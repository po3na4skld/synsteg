// Benchmark harness for Conversation-Steganography (github.com/nethical6/conversation-steganography @ 863a96e).
//
//	bench codec -config default -lengths 128,256,512 -runs 30 -out results/capacity_default.jsonl
//	bench codec -config u32 -lengths 128 -runs 400 -decode=false -out results/corpus_u32.jsonl
//	bench chain -config u1024 -sends 6 -out results/chain_u1024.jsonl
//
// "codec" drives the library's raw GenerativeCodec (framed payload of L random bits, no encryption, no chaining).
// "chain" drives the full protocol (ConversationChain: SIV encryption, packing, rolling transcript) with two
// independent participants alternating, as cmd/conversation-stenography/simulate.go does.
// Paths are resolved against $SYNSTEG_ROOT (default: current directory). Output files are appended to and
// already finished runs are skipped, so an interrupted job can be restarted with the same command.
package main

import (
	"bufio"
	"context"
	cs "conversationstenography"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// The three configurations of the original benchmark; every other setting comes from the example config.
var configs = map[string]struct {
	coding string
	topN   int
}{
	"default": {"arithmetic", 8},
	"u32":     {"uniform", 32},
	"u1024":   {"uniform", 1024},
}

type exampleConfig struct {
	Prompt            string  `json:"prompt"`
	TopN              int     `json:"top_n"`
	Coding            string  `json:"coding"`
	Temperature       float64 `json:"temperature"`
	FinishTokens      int     `json:"finish_tokens"`
	ChainSystem       string  `json:"chain_system"`
	StrictStyle       bool    `json:"strict_style"`
	CandidatePool     int     `json:"candidate_pool"`
	RefreshSentences  bool    `json:"refresh_sentences"`
	CarrierTrials     int     `json:"carrier_trials"`
	NaturalnessSlack  float64 `json:"naturalness_slack"`
	SemanticJudge     bool    `json:"semantic_judge"`
	SemanticThreshold float64 `json:"semantic_threshold"`
	LengthBias        float64 `json:"length_bias"`
	Model             string  `json:"model"`
	Revision          string  `json:"revision"`
}

func root(p string) string {
	if filepath.IsAbs(p) {
		return p
	}
	r := os.Getenv("SYNSTEG_ROOT")
	if r == "" {
		r = "."
	}
	return filepath.Join(r, p)
}

func die(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 || (os.Args[1] != "codec" && os.Args[1] != "chain") {
		die(fmt.Errorf("usage: bench codec|chain [flags]"))
	}
	mode := os.Args[1]
	fs := flag.NewFlagSet(mode, flag.ExitOnError)
	const clone = "experiments/conversation_bench/conversation-steganography"
	cfgName := fs.String("config", "default", "default | u32 | u1024")
	example := fs.String("example", clone+"/conversation-stenography.example.json", "example config (relative to $SYNSTEG_ROOT)")
	backend := fs.String("backend", clone+"/python/hf_model.py", "model process script")
	python := fs.String("python", "experiments/ai_detectors/.venv/bin/python", "Python interpreter")
	lengths := fs.String("lengths", "128,256,512", "codec: payload lengths in bits (multiples of 8)")
	runs := fs.Int("runs", 30, "codec: runs per length")
	decode := fs.Bool("decode", true, "codec: decode and check every stegotext")
	sends := fs.Int("sends", 6, "chain: number of sends")
	chainBits := fs.Int("bits", 128, "chain: plaintext bits per send")
	seed := fs.Uint64("seed", 20260815, "base seed for payloads (and chain key)")
	out := fs.String("out", "", "output JSONL (appended)")
	fs.Parse(os.Args[2:])
	sel, ok := configs[*cfgName]
	if !ok || *out == "" {
		die(fmt.Errorf("unknown -config %q or missing -out", *cfgName))
	}

	raw, err := os.ReadFile(root(*example))
	if err != nil {
		die(err)
	}
	var ex exampleConfig
	if err := json.Unmarshal(raw, &ex); err != nil {
		die(err)
	}
	gcfg := cs.GenerativeConfig{
		Prompt: ex.Prompt, TopN: sel.topN, Coding: sel.coding, Temperature: ex.Temperature, FinishTokens: ex.FinishTokens,
		ChainSystem: ex.ChainSystem, StrictStyle: ex.StrictStyle, CandidatePool: ex.CandidatePool,
		RefreshSentences: ex.RefreshSentences, CarrierTrials: ex.CarrierTrials, NaturalnessSlack: ex.NaturalnessSlack,
		SemanticJudge: ex.SemanticJudge, SemanticThreshold: ex.SemanticThreshold, LengthBias: ex.LengthBias,
	}

	ctx := context.Background()
	// Same backend arguments as simulate.go uses for the transformers runtime.
	model, err := cs.NewProcessModel(ctx, root(*python), root(*backend), "--model", ex.Model, "--revision", ex.Revision,
		"--device", "cpu", "--dtype", "float32")
	if err != nil {
		die(err)
	}
	defer model.Close()

	done := map[string]bool{}
	if f, err := os.Open(root(*out)); err == nil {
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1<<20), 1<<26)
		for sc.Scan() {
			var r struct{ Key string }
			if json.Unmarshal(sc.Bytes(), &r) == nil {
				done[r.Key] = true
			}
		}
		f.Close()
	}
	os.MkdirAll(filepath.Dir(root(*out)), 0o755)
	f, err := os.OpenFile(root(*out), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		die(err)
	}
	defer f.Close()
	emit := func(rec map[string]any) {
		b, _ := json.Marshal(rec)
		f.Write(append(b, '\n'))
		f.Sync()
	}
	common := map[string]any{"config": *cfgName, "coding": sel.coding, "top_n": sel.topN, "fingerprint": model.Fingerprint()}

	if mode == "codec" {
		runCodec(ctx, model, gcfg, *lengths, *runs, *decode, *seed, done, common, emit)
	} else {
		runChain(ctx, model, gcfg, *sends, *chainBits, *seed, done, common, emit)
	}
}

func rng(seed uint64, parts ...string) *rand.Rand {
	h := sha256.Sum256([]byte(strings.Join(parts, "/")))
	return rand.New(rand.NewPCG(seed, binary.BigEndian.Uint64(h[:8])))
}

func randomBytes(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.Uint32())
	}
	return b
}

func merge(a map[string]any, b map[string]any) map[string]any {
	for k, v := range a {
		b[k] = v
	}
	return b
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func runCodec(ctx context.Context, model *cs.ProcessModel, cfg cs.GenerativeConfig, lengths string, runs int, decode bool,
	seed uint64, done map[string]bool, common map[string]any, emit func(map[string]any)) {
	codec, err := cs.NewGenerativeCodec(model, cfg)
	if err != nil {
		die(err)
	}
	for _, ls := range strings.Split(lengths, ",") {
		L, err := strconv.Atoi(strings.TrimSpace(ls))
		if err != nil || L%8 != 0 {
			die(fmt.Errorf("bad length %q", ls))
		}
		for run := 0; run < runs; run++ {
			key := fmt.Sprintf("codec/%d/%d", L, run)
			if done[key] {
				continue
			}
			payload := randomBytes(rng(seed, common["config"].(string), key), L/8)
			t0 := time.Now()
			text, m, encErr := codec.EncodeWithMetrics(ctx, payload)
			encSec := time.Since(t0).Seconds()
			rec := map[string]any{"Key": key, "L": L, "run": run, "payload_hex": fmt.Sprintf("%x", payload),
				"encode_s": encSec, "encode_error": errString(encErr)}
			if encErr == nil {
				toks, _ := model.Tokenize(ctx, text)
				words := len(strings.Fields(text))
				rec["stegotext"] = text
				rec["words"] = words
				rec["gpt2_tokens"] = len(toks)
				rec["data_tokens"] = m.DataTokens
				rec["finish_tokens"] = m.FinishTokens
				rec["chars"] = m.VisibleCharacters
				rec["mean_logit_regret"] = m.MeanLogitRegret
				rec["worst_logit_regret"] = m.WorstLogitRegret
				if words > 0 {
					rec["bpw"] = float64(L) / float64(words)
				}
				if decode {
					t1 := time.Now()
					got, decErr := codec.Decode(ctx, text)
					rec["decode_s"] = time.Since(t1).Seconds()
					rec["decode_error"] = errString(decErr)
					rec["exact"] = decErr == nil && string(got) == string(payload)
				}
			}
			emit(merge(common, rec))
			fmt.Printf("%s L=%d run=%d enc=%.2fs words=%v exact=%v err=%q\n", common["config"], L, run, encSec,
				rec["words"], rec["exact"], errString(encErr))
		}
	}
}

func runChain(ctx context.Context, model *cs.ProcessModel, cfg cs.GenerativeConfig, sends, bits int, seed uint64,
	done map[string]bool, common map[string]any, emit func(map[string]any)) {
	if len(done) > 0 {
		die(fmt.Errorf("chain runs depend on the whole transcript; delete the output file to rerun"))
	}
	r := rng(seed, common["config"].(string), "chain")
	key := randomBytes(r, 32)
	const conversation = "bench-simulation"
	alice, err := cs.NewConversationChain(model, key, conversation, cfg)
	if err != nil {
		die(err)
	}
	bob, err := cs.NewConversationChain(model, key, conversation, cfg)
	if err != nil {
		die(err)
	}
	names := [2]string{"Alice", "Bob"}
	chains := [2]*cs.ConversationChain{alice, bob}
	for i := 0; i < sends; i++ {
		from, sender, receiver := names[i%2], chains[i%2], chains[1-i%2]
		plaintext := randomBytes(r, bits/8)
		budget, _ := sender.EncodingBudget(plaintext)
		t0 := time.Now()
		record, sendErr := sender.Send(ctx, from, plaintext)
		rec := map[string]any{"Key": fmt.Sprintf("chain/%d", i), "send": i, "from": from, "bits": bits,
			"plaintext_hex": fmt.Sprintf("%x", plaintext), "hidden_bytes": budget.TotalHiddenBytes,
			"encode_s": time.Since(t0).Seconds(), "send_error": errString(sendErr), "completed": sendErr == nil}
		if sendErr == nil {
			rec["stegotext"] = record.Encrypted
			rec["words"] = len(strings.Fields(record.Encrypted))
			t1 := time.Now()
			got, _, recvErr := receiver.Receive(ctx, from, record.Encrypted)
			rec["decode_s"] = time.Since(t1).Seconds()
			rec["receive_error"] = errString(recvErr)
			rec["decoded"] = recvErr == nil && string(got) == string(plaintext)
			if recvErr != nil {
				// Keep the receiver's public transcript in sync so later sends are measured independently.
				if err := receiver.RestorePublic([]cs.ChainRecord{record}); err != nil {
					rec["resync_error"] = err.Error()
				}
			}
		}
		emit(merge(common, rec))
		fmt.Printf("%s send=%d completed=%v decoded=%v err=%q %q\n", common["config"], i, rec["completed"], rec["decoded"],
			rec["send_error"], rec["receive_error"])
	}
}

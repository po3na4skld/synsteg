#!/bin/sh
# Fetch the two third-party code bases used in the experiments at the exact commits, with the local patches.
#   experiments/steganalysis/TextSteganalysis               yjs1224/TextSteganalysis @ ff773a4 (steganalysis code of DeepStego, protocol A')
#   experiments/conversation_bench/conversation-steganography nethical6/conversation-steganography @ 863a96e (generative baseline, subsection 3.6)
set -e
cd "$(dirname "$0")"
fetch() { # dir url commit patch
  if [ ! -d "$1/.git" ]; then git clone -q "$2" "$1"; fi
  git -C "$1" checkout -q "$3"
  git -C "$1" apply --check "$PWD/$4" 2>/dev/null && git -C "$1" apply "$PWD/$4" || echo "$4 already applied or not applicable: check $1"
}
fetch steganalysis/TextSteganalysis https://github.com/yjs1224/TextSteganalysis.git ff773a4 steganalysis/textsteganalysis_local.patch
fetch conversation_bench/conversation-steganography https://github.com/nethical6/conversation-steganography.git 863a96e conversation_bench/hf_model_no_grad.patch
echo "third-party code ready"

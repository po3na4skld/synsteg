# Synonym Steganography Bot


# local_runner.py Script

This Python script encodes and decodes a hidden message within a text-based container file using steganography. 
It creates a secure, random message of specified length and encodes it using parameters such as bits per word, additional bits, and container buffer.

## Features
- Generates a random secret message of specified bit length.
- Embeds the message within a text container using customizable encoding parameters.
- Provides logging for encoding/decoding steps and saves reports as JSON files in the specified output path.
- Ensures security and message fitting within the container constraints.

## Requirements
- Python 3.10.10 or higher

## Installation
1. Clone the repository.
3. Install required modules and dependencies.

```bash
pip install -r requirements.txt
```

## Usage

Run the script with the following arguments:

```bash
python local_runner.py -container_path <path_to_container> -openai_key <your_openai_key> -bits_per_word <bits>
```

### Arguments

- **`-container_path`**: Required. Path to the `.txt` file containing the container message for encoding.
- **`-output_path`**: Optional. Directory path for output artifacts (default is `artifacts/`).
- **`-message_length`**: Optional. Length of the random secret message in bits. Choose from `[128, 256, 512]`.
- **`-bits_per_word`**: Required. Specifies the number of bits to encode per word.
- **`-additional_bits`**: Optional. Number of additional bits to encode per word (default is `0`).
- **`-config_path`**: Optional. Path to the configuration file containing hyperparameters (default is `config.json`).
- **`-openai_key`**: Required. Your OpenAI API key for processing.

### Example

```bash
python local_runner.py -container_path container.txt -output_path artifacts -message_length 256 -bits_per_word 5 -additional_bits 1 -openai_key your_openai_api_key
```

## Output
- Logs key processing steps to the console.
- Generates a JSON report containing encoding/decoding details in `artifacts/`.

## Logging and Debugging
The script uses a logger to display important messages, warnings, and errors. 
To modify the logging configuration, edit the logger in `utils/logger.py`.

## Constraints
- `bits_per_word` should not exceed `MAX_BITS_PER_WORD` from `utils.constants`.
- The sum of `bits_per_word` and `additional_bits` should be within limits defined by `MAX_ADDITIONAL_BITS_MULTIPLIER`.

## Example Workflow
1. **Container Loading**: Loads a `.txt` container file.
2. **Random Message Generation**: Creates a random secret message.
3. **Encoding**: Encodes the message in the container using specified parameters.
4. **Decoding**: Decodes the message to validate the encoding.
5. **Reporting**: Saves results, including time and usage reports, to a JSON file.

## Reports file (Dataset) link

(dataset link)[https://drive.google.com/drive/folders/1e0kVpgAABiMc7sK5o6OjlTLv01bFTINB?usp=sharing]

# Pakhomov GSC

**Pakhomov GSC** (Generative Seed Compression) is a lossless compression tool based on deterministic pseudorandom generation. It encodes a file as a seed that reproduces the same output via a seeded PRNG. This makes decompression extremely lightweight and fast — all you need is the seed and the PRNG logic. **PGSZ** is the file extension and always weights 16 bytes!

---

## Features

* Deterministic compression using `xoshiro1024++`
* CPU and OpenCL (GPU) accelerated compression
* Multi-threaded CPU seed search
* Bit-accurate encoding (for truly minimal encodings)
* Outputs seed + bit length (instead of entire byte stream)
* Customizable chunk size for GPU processing

---

## Build

Requires C++20, and OpenCL headers.

```bash
git clone https://github.com/vladisrael/pakhomov-gsc.git
cd pakhomov-gsc
sh compile.sh
```

---

## Usage

### Compression

#### CPU Version:

```bash
./pakhomov-gsc compress <input_file>
```

* `input_file` goes in output file with `.pgsz` extension.
* Compresses the file using multi-threaded CPU search.

#### OpenCL Version (GPU):

```bash
./pakhomov-gsc compress-cl <input_file> [chunk_size]
```

* Uses OpenCL to search for a matching seed in parallel.
* `input_file` goes in output file with `.pgsz` extension.
* `chunk_size` defines the number of threads per batch (default: `1 000 000 000`).

---

### Decompression

```bash
./pakhomov-gsc decompress <compressed_file>
```

* Reconstructs the original file from the seed and bit count.
* `compressed_file` goes in output file removing the `.pgsz` extension.

---

## Compressed File Format

All compressed files are binary files containing:

```
<seed> <bit_length>
```

Should always be 16 bytes. So it kinda nukes all compression. (If actually convenient)

---


## Use Case

This project is intended for research, theoretical exploration, and experimentation in data entropy, compression theory, and algorithmic generation. It is not a practical compression tool.

---

## Performance Tips

* Use OpenCL if you have a modern GPU to greatly accelerate seed search.
* Try adjusting the `chunk_size` for better performance depending on GPU memory.

---

## Author

**Vladimir Alexandre Pakhomov**

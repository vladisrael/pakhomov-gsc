#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <iterator>
#include "xoshiro1024pp.hpp"
#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/opencl.hpp>

using namespace std;

// --- Config ---
const int THREAD_COUNT = thread::hardware_concurrency(); // use all cores
// ---------------

mutex output_mutex;
atomic<bool> found(false);
uint64_t found_seed = 0;

// Read entire binary file
vector<unsigned char> read_file(const string& filename) {
    ifstream file(filename, ios::binary);
    return vector<unsigned char>((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

// Write binary file
void write_file(const string& filename, const vector<unsigned char>& data) {
    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// Convert bytes -> packed uint64_t bit vector
vector<uint64_t> file_to_bits_u64(const vector<unsigned char>& data, size_t& out_bit_count) {
    size_t bit_count = data.size() * 8;
    vector<uint64_t> bits((bit_count + 63) / 64, 0);

    for (size_t i = 0; i < bit_count; ++i) {
        if (data[i / 8] & (1 << (7 - (i % 8)))) {
            bits[i / 64] |= (uint64_t(1) << (63 - (i % 64)));
        }
    }

    // Mask out garbage bits in last word
    if (bit_count % 64 != 0) {
        int valid_bits = bit_count % 64;
        uint64_t mask = ((uint64_t)1 << valid_bits) - 1;
        bits.back() &= (mask << (64 - valid_bits));
    }

    out_bit_count = bit_count;
    return bits;
}

// Generate PRNG bits into uint64_t vector
vector<uint64_t> generate_bits_u64(uint64_t seed, size_t bit_count) {
    xoshiro1024pp rng(seed);
    size_t word_count = (bit_count + 63) / 64;
    vector<uint64_t> bits(word_count);

    for (size_t i = 0; i < word_count; ++i) {
        bits[i] = rng.next(); // One 64-bit word per call
    }

    // Mask unused bits in the last word (optional but safer)
    if (bit_count % 64 != 0) {
        int valid_bits = bit_count % 64;
        uint64_t mask = ((uint64_t)1 << valid_bits) - 1;
        mask <<= (64 - valid_bits);
        bits[word_count - 1] &= mask;
    }

    return bits;
}

// Convert uint64_t bit vector -> bytes
vector<unsigned char> bits_to_bytes_u64(const vector<uint64_t>& bits, size_t bit_count) {
    vector<unsigned char> bytes((bit_count + 7) / 8, 0);

    for (size_t i = 0; i < bit_count; ++i) {
        if (bits[i / 64] & (uint64_t(1) << (63 - (i % 64)))) {
            bytes[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
    return bytes;
}

// Worker thread function
void search_seed(const vector<uint64_t>& target_bits, size_t bit_count, uint64_t start_seed, uint64_t step) {
    uint64_t seed = start_seed;
    while (!found) {
        vector<uint64_t> gen = generate_bits_u64(seed, bit_count);
        if (gen == target_bits) {
            lock_guard<mutex> lock(output_mutex);
            if (!found) {
                found = true;
                found_seed = seed;
            }
            break;
        }
        seed += step;
        if (seed % 1'000'000 == 0) {
            std::cout << "Seed progress: " << seed << std::endl;
        }
    }
}

// Helper to read kernel source from file
std::string read_kernel(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Failed to open kernel file.");
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

// Compress: find seed
void compress(const string& input_file, const string& output_file) {
    vector<unsigned char> data = read_file(input_file);
    size_t bit_count;
    vector<uint64_t> target_bits = file_to_bits_u64(data, bit_count);

    cout << "Target: " << bit_count << " bits. Using " << THREAD_COUNT << " threads." << endl;

    vector<thread> threads;
    for (int i = 0; i < THREAD_COUNT; ++i) {
        threads.emplace_back(search_seed, cref(target_bits), bit_count, i, THREAD_COUNT);
    }

    for (auto& t : threads) {
        t.join();
    }

    if (found) {
        std::ofstream out(output_file, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&found_seed), sizeof(found_seed));
        out.write(reinterpret_cast<const char*>(&bit_count), sizeof(bit_count));
        out.close();
        cout << "Seed found: " << found_seed << endl;
        cout << "Saved to: " << output_file << endl;
    } else {
        cout << "No seed found (unexpected)." << endl;
    }
}

// Compress: find seed with OpenCL
void compress_cl(const std::string& input_file, const std::string& output_file, size_t chunk_size = 1'000'000'000) {
    std::vector<unsigned char> data = read_file(input_file);
    size_t bit_count;
    std::vector<uint64_t> target_bits = file_to_bits_u64(data, bit_count);

    std::cout << "Target: " << bit_count << " bits. Using OpenCL GPU acceleration." << std::endl;
    std::cout << "Using chunk size: " << chunk_size << " threads per batch.\n" << std::endl;

    try {
        // 1. Get OpenCL platforms and devices
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("No OpenCL platforms found");

        cl::Platform platform = platforms[0];
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) throw std::runtime_error("No OpenCL GPU devices found");

        cl::Device device = devices[0];
        cl::Context context(device);
        cl::CommandQueue queue(context, device);

        // 2. Read and build kernel
        std::string kernel_source = read_kernel("pakhomov-gsc.cl");
        cl::Program program(context, kernel_source);
        try {
            program.build("-cl-std=CL2.0");
        } catch (...) {
            std::cerr << "Kernel build error:\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
            throw;
        }

        cl::Kernel kernel(program, "seed_search");

        // 3. Create buffers
        cl::Buffer buffer_target_bits(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      sizeof(uint64_t) * target_bits.size(), target_bits.data());
        cl::Buffer buffer_out_seed(context, CL_MEM_WRITE_ONLY, sizeof(uint64_t));
        cl::Buffer buffer_found_flag(context, CL_MEM_READ_WRITE, sizeof(int));

        uint64_t start_seed = 0;
        int found_flag = 0;

        while (true) {
            // 4. Set kernel arguments for this chunk
            kernel.setArg(0, buffer_out_seed);        // result
            kernel.setArg(1, buffer_target_bits);     // target
            kernel.setArg(2, (cl_ulong)bit_count);    // bit_count
            kernel.setArg(3, (cl_ulong)start_seed);   // start_seed offset
            kernel.setArg(4, buffer_found_flag);      // found_flag

            // Reset flag
            found_flag = 0;
            queue.enqueueWriteBuffer(buffer_found_flag, CL_TRUE, 0, sizeof(int), &found_flag);

            // 5. Launch kernel
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(chunk_size));
            queue.finish();

            // 6. Check if a match was found
            queue.enqueueReadBuffer(buffer_found_flag, CL_TRUE, 0, sizeof(int), &found_flag);
            if (found_flag != 0) {
                uint64_t found_seed;
                queue.enqueueReadBuffer(buffer_out_seed, CL_TRUE, 0, sizeof(uint64_t), &found_seed);

                std::ofstream out(output_file, std::ios::binary);
                out.write(reinterpret_cast<const char*>(&found_seed), sizeof(found_seed));
                out.write(reinterpret_cast<const char*>(&bit_count), sizeof(bit_count));
                out.close();

                std::cout << "Seed found: " << found_seed << std::endl;
                std::cout << "Saved to: " << output_file << std::endl;
                break;
            }

            // 7. Move to next chunk
            start_seed += chunk_size;
            std::cout << "Seed progress: " << start_seed << std::endl;

            // Optional: reduce CPU usage a bit
            //std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    } catch (const std::exception& e) {
        std::cerr << "OpenCL error: " << e.what() << std::endl;
    }
}

// Decompress: regen file
void decompress(const string& input_file, const string& output_file) {
    std::ifstream in(input_file, std::ios::binary);
    uint64_t seed;
    size_t bit_count;
    in.read(reinterpret_cast<char*>(&seed), sizeof(seed));
    in.read(reinterpret_cast<char*>(&bit_count), sizeof(bit_count));

    vector<uint64_t> bits = generate_bits_u64(seed, bit_count);
    vector<unsigned char> bytes = bits_to_bytes_u64(bits, bit_count);
    write_file(output_file, bytes);

    cout << "Decompressed file with seed " << seed << " (" << bit_count << " bits)." << endl;
}

// Main
int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Pakhomov GSC (Generative Seed Compression) (2025)\n";
        cout << "Usage:\n";
        cout << "  " << argv[0] << " compress <input_file> [compressed_file]\n";
        cout << "  " << argv[0] << " compress-cl <input_file> [compressed_file] [chunk_size]\n";
        cout << "  " << argv[0] << " decompress <compressed_file> [output_file]\n";
        return 1;
    }

    string command = argv[1];
    if (command == "compress") {
        string input = argv[2];
        string output = (argc > 3) ? argv[3] : (argv[2] + string(".bin"));
        compress(input, output);
    } else if (command == "compress-cl") {
        string input = argv[2];
        string output = (argc > 3) ? argv[3] : (argv[2] + string(".bin"));
        size_t chunk_size = (argc > 4) ? std::stoull(argv[4]) : 1'000'000'000;
        compress_cl(input, output, chunk_size);
    } else if (command == "decompress") {
        string input = argv[2];
        string output = (argc > 3) ? argv[3] : "output.bin";
        decompress(input, output);
    } else {
        cout << "Unknown command: " << command << endl;
        return 1;
    }

    return 0;
}

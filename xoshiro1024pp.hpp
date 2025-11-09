#pragma once
#include <cstdint>
#include <array>

// xoshiro1024++ 1.0 - Public domain, by David Blackman & Sebastiano Vigna
// C++ class wrapper version (portable, no global state)
// Reference: http://prng.di.unimi.it/

class xoshiro1024pp {
public:
    using state_t = std::array<uint64_t, 16>;

    explicit xoshiro1024pp(uint64_t seed_val = 1) {
        seed_splitmix64(seed_val);
    }

    // Generate next 64-bit random value
    uint64_t next() {
        const int q = p;
        const uint64_t s0 = s[p = (p + 1) & 15];
        uint64_t s15 = s[q];
        const uint64_t result = rotl(s0 + s15, 23) + s15;

        s15 ^= s0;
        s[q] = rotl(s0, 25) ^ s15 ^ (s15 << 27);
        s[p] = rotl(s15, 36);

        return result;
    }

    // Jump equivalent to 2^512 calls to next()
    void jump() {
        static const uint64_t JUMP[] = {
            0x931197d8e3177f17ULL, 0xb59422e0b9138c5fULL, 0xf06a6afb49d668bbULL,
            0xacb8a6412c8a1401ULL, 0x12304ec85f0b3468ULL, 0xb7dfe7079209891eULL,
            0x405b7eec77d9eb14ULL, 0x34ead68280c44e4aULL, 0xe0e4ba3e0ac9e366ULL,
            0x8f46eda8348905b7ULL, 0x328bf4dbad90d6ffULL, 0xc8fd6fb31c9effc3ULL,
            0xe899d452d4b67652ULL, 0x45f387286ade3205ULL, 0x03864f454a8920bdULL,
            0xa68fa28725b1b384ULL
        };
        uint64_t t[16] = {0};

        for (int i = 0; i < 16; ++i)
            for (int b = 0; b < 64; ++b) {
                if (JUMP[i] & (1ULL << b))
                    for (int j = 0; j < 16; ++j)
                        t[j] ^= s[(j + p) & 15];
                next();
            }

        for (int i = 0; i < 16; ++i)
            s[(i + p) & 15] = t[i];
    }

    // Long jump equivalent to 2^768 calls to next()
    void long_jump() {
        static const uint64_t LONG_JUMP[] = {
            0x7374156360bbf00fULL, 0x4630c2efa3b3c1f6ULL, 0x6654183a892786b1ULL,
            0x94f7bfcbfb0f1661ULL, 0x27d8243d3d13eb2dULL, 0x9701730f3dfb300fULL,
            0x2f293baae6f604adULL, 0xa661831cb60cd8b6ULL, 0x68280c77d9fe008cULL,
            0x50554160f5ba9459ULL, 0x2fc20b17ec7b2a9aULL, 0x49189bbdc8ec9f8fULL,
            0x92a65bca41852cc1ULL, 0xf46820dd0509c12aULL, 0x52b00c35fbf92185ULL,
            0x1e5b3b7f589e03c1ULL
        };
        uint64_t t[16] = {0};

        for (int i = 0; i < 16; ++i)
            for (int b = 0; b < 64; ++b) {
                if (LONG_JUMP[i] & (1ULL << b))
                    for (int j = 0; j < 16; ++j)
                        t[j] ^= s[(j + p) & 15];
                next();
            }

        for (int i = 0; i < 16; ++i)
            s[(i + p) & 15] = t[i];
    }

private:
    state_t s{};
    int p = 0;

    static uint64_t rotl(uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    void seed_splitmix64(uint64_t seed_val) {
        uint64_t z = seed_val;
        for (int i = 0; i < 16; ++i) {
            s[i] = splitmix64(z);
            z += 0x9e3779b97f4a7c15ULL;
        }
        p = 0;
    }

    static uint64_t splitmix64(uint64_t& z) {
        uint64_t result = z;
        result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
        result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
        return result ^ (result >> 31);
    }
};

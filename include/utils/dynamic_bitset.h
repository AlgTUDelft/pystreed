#include "base.h"
#include <climits>

namespace STreeD {

	struct DynamicBitSet {
		using BaseType = unsigned long;
		static constexpr size_t BITS_PER_ELEMENT = sizeof(BaseType) * CHAR_BIT;
		static constexpr size_t BYTES_PER_ELEMENT = sizeof(BaseType);

		BaseType* bitset;
		size_t elements;

		DynamicBitSet(size_t size) {
			elements = (size - 1) / BITS_PER_ELEMENT + 1;
			bitset = new BaseType[elements];
			std::fill(bitset, bitset + elements, 0);
		}

		DynamicBitSet() : DynamicBitSet(1) {}

		~DynamicBitSet() {
			delete[] bitset;
		}

		DynamicBitSet(const DynamicBitSet& other) : elements(other.elements) {
			bitset = new BaseType[elements];
			std::memcpy(bitset, other.bitset, elements * BYTES_PER_ELEMENT);
		}

		DynamicBitSet& operator=(const DynamicBitSet& other) {
			if (this == &other)
				return *this;

			elements = other.elements;
			BaseType* new_bitset = new BaseType[elements];
			std::memcpy(new_bitset, other.bitset, elements * BYTES_PER_ELEMENT);
			delete[] bitset;
			bitset = new_bitset;
			return *this;
		}

		bool operator==(const DynamicBitSet& other) const {
			runtime_assert(elements == other.elements);
			for (size_t i = 0; i < elements; i++) {
				if (bitset[i] != other.bitset[i]) return false;
			}
			return true;
		}

		inline bool operator!=(const DynamicBitSet& other) const {
			return !((*this) == other);
		}

		void SetBit(size_t index) {
			size_t element = index / BITS_PER_ELEMENT;
			size_t bit_index = index % BITS_PER_ELEMENT;
			runtime_assert(element < elements);
			bitset[element] |= 1UL << bit_index;
		}

		void ClearBit(size_t index) {
			size_t element = index / BITS_PER_ELEMENT;
			size_t bit_index = index % BITS_PER_ELEMENT;
			runtime_assert(element < elements);
			bitset[element] &= 1UL << bit_index;
		}

		void ToggleBit(size_t index) {
			size_t element = index / BITS_PER_ELEMENT;
			size_t bit_index = index % BITS_PER_ELEMENT;
			runtime_assert(element < elements);
			bitset[element] ^= 1UL << bit_index;
		}

	};

}

namespace std {

	template <>
	struct hash<STreeD::DynamicBitSet> {

		//adapted from https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector
		size_t operator()(const STreeD::DynamicBitSet& bitset) const {
			using std::size_t;
			using std::hash;
			size_t seed = 0;
			for (int i = 0; i < bitset.elements; i++) {
				seed ^= hash<STreeD::DynamicBitSet::BaseType>()(bitset.bitset[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
			}
			return seed;
		}

	};

}
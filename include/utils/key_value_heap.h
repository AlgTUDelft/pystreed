//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirovic
//https://bitbucket.org/EmirD/murtree/src/master/code/MurTree/Data%20Structures/key_value_heap.h

#ifndef KEY_VALUE_HEAP
#define KEY_VALUE_HEAP

#include <vector>

/*
A heap class where the keys range from [0, ..., n-1] and the values are nonnegative floating points.
The heap can be queried to return key with the maximum value, and certain keys can be (temporarily) removed/readded as necessary
It allows increasing/decreasing the values of its entries
*/

//should maybe check for numerical stability...
class KeyValueHeap {
public:
	KeyValueHeap(int num_entries); //create a heap with keys [0, ..., num_entries-1] and values all set to zero. Internally 'reserve' entry spaces will be allocated.
	//no need for now ~KeyValueHeap();

	void Increment(int key_id, double increment); //increments the value of the element of 'key_id' by increment. O(logn) worst case, but average case might be better.
	int PopMax(); //returns the key_id with the highest value, and removes the key from the heap. O(logn)
	void Remove(int key_id); //removes the entry with key 'key_id' (temporarily) from the heap. Its value remains recorded internally and is available upon readding 'key_id' to the heap. The value can still be subjected to 'DivideValues'. O(logn)
	void Readd(int key_id); //readd the entry with key 'key_id' to the heap. Assumes it was present in heap before and that it is in present in the current state. Its value is the previous value used before Remove(key_id) was called.  O(logn)
	void Grow(); //increases the Size of the heap by one. The key will have zero assigned to its value.
	void DivideValues(double divisor); //divides all the values in the heap by 'divisor'. This will affect the values of keys that have been removed. O(n)

	double GetKeyValue(int key_id) const;
	bool IsKeyPresent(int key_id) const;
	double ReadMaxValue() const; //read the key with the highest value without removing it. O(1)
	int Size() const;  //returns the Size of the heap. O(1)
	bool Empty() const; //return if the heap is empty, i.e. Size() == 0.
	int MaxSize() const; //returns the number of elements the heap can support without needing to call Grow(). O(1).

private:
	std::vector<double> values_; //contains the values stored as a heap
	std::vector<int> map_key_to_position_; //[i] shows the location of the value of the key i in the values_ array
	std::vector<int> map_position_to_key_; //[i] shows which key is associated with values_[i]
	int end_position_; //the index past the last element in the heap

	void SwapPositions(int i, int j);
	void SiftUp(int i);
	void SiftDown(int i);

	int GetParent(int i) const;
	int GetLeftChild(int i) const;
	int GetRightChild(int i) const;
	int GetLargestChild(int i) const;

	bool IsHeap(int position) const;
	bool IsHeapLocally(int position) const;
	bool IsLeaf(int position) const;
	bool IsSmallerThanLeftChild(int position) const;
	bool IsSmallerThanRightChild(int position) const; //note that it returns false if there is no right child
	bool CheckSizeConsistency() const;

	bool IsMaxValue(double) const;

	bool AreDataStructuresOfSameSize() const;
};

#endif // !KEY_VALUE_HEAP
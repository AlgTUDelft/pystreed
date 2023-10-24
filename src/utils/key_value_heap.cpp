//Distributed under the MIT license, see License.txt
//Copyright © 2022 Emir Demirovic
//https://bitbucket.org/EmirD/murtree/src/master/code/MurTree/Data%20Structures/key_value_heap.cpp

#include "utils/key_value_heap.h"

#include <assert.h>

KeyValueHeap::KeyValueHeap(int num_entries)
	:values_(num_entries, 0.0), map_key_to_position_(num_entries), map_position_to_key_(num_entries), end_position_(num_entries) {
	for (int i = 0; i < num_entries; i++) {
		map_position_to_key_[i] = i;
		map_key_to_position_[i] = i;
	}
}

void KeyValueHeap::Increment(int key_id, double increment) {
	int i = map_key_to_position_[key_id];
	values_[i] += increment;
	if (i < end_position_) { //do not sift up values that are not in the heap 
		SiftUp(i);
	}
}

int KeyValueHeap::PopMax() {
	assert(end_position_ > 0); //must be something in the heap
	assert(IsHeap(0));

	int best_key = map_position_to_key_[0];
	Remove(best_key);
	return best_key;
}

void KeyValueHeap::Remove(int key_id) {
	//place the key at the end of the heap, decrement the heap, and sift down
	int position = map_key_to_position_[key_id];
	SwapPositions(position, end_position_ - 1);
	end_position_--;
	//if (end_position_ > 1) 
	if (end_position_ > 1 && position < end_position_) //no need to sift for empty or one-node heaps, nor do we do this in case the element removed was anyway the last element
	{
		SiftDown(position);
	}

	assert(end_position_ == 0 || IsHeap(0));
}

void KeyValueHeap::Readd(int key_id) {
	assert(end_position_ < values_.size());

	//key_id is somewhere in the position [end_position, max_size-1]
	//place the key at the end of the heap, increase end_position, and sift up
	int key_position = map_key_to_position_[key_id];
	SwapPositions(key_position, end_position_);
	end_position_++;
	SiftUp(end_position_ - 1);
}

void KeyValueHeap::Grow() {
	assert(CheckSizeConsistency());

	int new_key = int(values_.size());
	values_.push_back(0);
	map_key_to_position_.push_back(new_key);//initially played at the very end, wil be swapped
	map_position_to_key_.push_back(new_key);
	SwapPositions(end_position_, new_key);
	end_position_++;
	SiftUp(end_position_ - 1);
}

void KeyValueHeap::DivideValues(double divisor) {
	for (double& val : values_) { val /= divisor; }
}

double KeyValueHeap::GetKeyValue(int key_id) const {
	int position = map_key_to_position_[key_id];
	return values_[position];
}

bool KeyValueHeap::IsKeyPresent(int key_id) const {
	return map_key_to_position_[key_id] < end_position_;
}

double KeyValueHeap::ReadMaxValue() const {
	assert(IsHeap(0));
	return values_[0];
}

int KeyValueHeap::Size() const {
	return end_position_;
}

bool KeyValueHeap::Empty() const {
	return Size() == 0;
}

int KeyValueHeap::MaxSize() const {
	assert(AreDataStructuresOfSameSize());
	return int(values_.size());
}

void KeyValueHeap::SwapPositions(int i, int j) {
	int key_i = map_position_to_key_[i];
	int key_j = map_position_to_key_[j];
	std::swap(values_[i], values_[j]);
	std::swap(map_position_to_key_[i], map_position_to_key_[j]);
	std::swap(map_key_to_position_[key_i], map_key_to_position_[key_j]);
}

void KeyValueHeap::SiftUp(int i) {
	if (i == 0) { return; } //base case at the root

	int parent_i = GetParent(i);
	if (values_[parent_i] >= values_[i]) { return; } //heap property satisfied, done

	SwapPositions(i, parent_i);
	SiftUp(parent_i); //note the new value has been pushed up into parent_id position
}

void KeyValueHeap::SiftDown(int i) {
	assert(i < end_position_);

	if (IsHeapLocally(i) == true) { return; }

	int largest_child = GetLargestChild(i);
	SwapPositions(i, largest_child);
	SiftDown(largest_child); //note the new value has been pushed down to largest_child position
}

int KeyValueHeap::GetParent(int i) const {
	assert(i > 0); //root has no parent
	int parent_id = (i - 1) / 2;
	assert(GetLeftChild(parent_id) == i || GetRightChild(parent_id) == i);
	return parent_id;
}

int KeyValueHeap::GetLeftChild(int i) const {
	int left_child = 2 * i + 1;
	return left_child;
}

int KeyValueHeap::GetRightChild(int i) const {
	int right_child = 2 * i + 2;
	if (right_child < end_position_) {
		return right_child;
	} else //there is no right child, note the tree is not necessarily a full binary tree
	{
		return -1;
	}
}

int KeyValueHeap::GetLargestChild(int i) const {
	assert(IsLeaf(i) == false);
	int max_child = GetLeftChild(i); //assume the left child is the largest
	int right_child = GetRightChild(i);
	if (right_child != -1 && values_[right_child] > values_[max_child]) //if there exists a right child, compare to it
	{
		max_child = right_child;
	}
	return max_child;
}

bool KeyValueHeap::IsHeap(int position) const {
	assert(position >= 0); //root is in position 0
	assert(position < end_position_); //cannot exceed the node count

	return true; //disabled temporarily

	if (IsHeapLocally(position) == false) { return false; }

	if (IsLeaf(position) == true) { return true; }

	int right_child = GetRightChild(position);
	return IsHeap(GetLeftChild(position)) && (right_child == -1 || IsHeap(right_child)); //check the 1) left child and the 2) right child if it exists
}

bool KeyValueHeap::IsHeapLocally(int position) const {
	if (IsLeaf(position) == true) { return true; } else if (IsSmallerThanLeftChild(position) || IsSmallerThanRightChild(position)) { return false; } else { return true; }
}

bool KeyValueHeap::IsLeaf(int position) const {
	return GetLeftChild(position) >= end_position_;
}

bool KeyValueHeap::IsSmallerThanLeftChild(int position) const {
	assert(IsLeaf(position) == false);
	return values_[position] < values_[GetLeftChild(position)];
}

bool KeyValueHeap::IsSmallerThanRightChild(int position) const {
	assert(IsLeaf(position) == false);
	int right_child = GetRightChild(position);
	return right_child != -1 && values_[position] < values_[right_child];
}

bool KeyValueHeap::CheckSizeConsistency() const {
	assert(values_.size() == map_key_to_position_.size());
	assert(values_.size() == map_position_to_key_.size());
	assert(end_position_ <= values_.size());
	return true;
}

bool KeyValueHeap::IsMaxValue(double reference) const {
	for (int i = 0; i < end_position_; i++) {
		double val = values_[i];
		if (reference < val) {
			assert(reference >= val);
		}
	}
	return true;
}

bool KeyValueHeap::AreDataStructuresOfSameSize() const {
	int number_of_entries = int(values_.size());
	assert(map_key_to_position_.size() == number_of_entries);
	assert(map_position_to_key_.size() == number_of_entries);
	assert(end_position_ <= number_of_entries);
	return true;
}

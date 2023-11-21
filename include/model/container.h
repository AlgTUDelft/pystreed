#pragma once
#include "base.h"
#include "model/node.h"

namespace STreeD {

	typedef std::make_signed_t<std::size_t> ssize_t;

    /*
    * Container for solutions of type Node<OT>
    * Only necessary for partially ordered Optimization tasks
    */
	template <class OT>
	class Container {
	public:

		Container<OT>() = default;
		Container<OT>(const Container<OT>& other) : solutions(other.solutions), max_num_nodes(other.max_num_nodes), max_depth(other.max_depth) {}

        // Add a node and use OT::Dominates to check for dominance and remove dominated solutions
		inline void Add(const Node<OT>& node) { InternalAdd<false>(node); }
		
        // Add a node and use OT::DominatesInv to check for reverse dominance and remove reverse dominated solutions
        inline void AddInv(const Node<OT>& node) { InternalAdd<true>(node); }

        // Add a node and use OT::DominatesD0 to check for root node dominance and remove dominated solutions
		inline void AddD0(OT* task, const Node<OT>& node) { InternalAddD0<false>(task, node); }
		
        // Add a node and use OT::DominatesD0Inv to check for root node reverse dominance and remove reverse dominated solutions
        inline void AddInvD0(OT* task, const Node<OT>& node) { InternalAddD0<true>(task, node); }

        // Add a solution using Add, except if a maximum size is exceeded. Then merge with the most similar solution
		inline void AddOrMerge(const Node<OT>& node, size_t max_size) { InternalAddOrMerge<false, false>(node, max_size); }
		
        // Add a solution using Add, except if a maximum size is exceeded. Then inverse merge with the most similar solution
        inline void AddOrInvMerge(const Node<OT>& node, size_t max_size) { InternalAddOrMerge<false, true>(node, max_size); }
		
        // Add a solution using AddInv, except if a maximum size is exceeded. Then merge with the most similar solution
        inline void AddInvOrMerge(const Node<OT>& node, size_t max_size) { InternalAddOrMerge<true, false>(node, max_size); }
		
        // Add a solution using AddInv, except if a maximum size is exceeded. Then inverse merge with the most similar solution
        inline void AddInvOrInvMerge(const Node<OT>& node, size_t max_size) { InternalAddOrMerge<true, true>(node, max_size); }
		
        // Iteratively add solutions
		inline void Add(const Container<OT>& container) { for (const auto& node : container.solutions) Add(node); }
		
        // Iteratively inverse add solutions
        inline void AddInv(const Container<OT>& container) { for (const auto& node : container.solutions) AddInv(node); }
		
        // Iteratively add solutions using AddD0
        inline void AddD0(OT* task, const Container<OT>& container) { for (const auto& node : container.solutions) AddD0(task, node); }

        // Iteratively AddOrMerge solutions
		inline void AddOrMerge(const Container<OT>& container, size_t max_size) { for (const auto& node : container.solutions) AddOrMerge(node, max_size); }
		
        // Iteratively AddOrInvMerge solutions
        inline void AddOrInvMerge(const Container<OT>& container, size_t max_size) { for (const auto& node : container.solutions) AddOrInvMerge(node, max_size); }
		
        // Iteratively AddInvOrMerge solutions
        inline void AddInvOrMerge(const Container<OT>& container, size_t max_size) { for (const auto& node : container.solutions) AddInvOrMerge(node, max_size); }
		
        // Iteratively AddInvOrInvMerge solutions
        inline void AddInvOrInvMerge(const Container<OT>& container, size_t max_size) { for (const auto& node : container.solutions) AddInvOrInvMerge(node, max_size); }

        // Get the number of solutions in this container
		inline size_t Size() const { return solutions.size(); }
		
        // Get the solutions in this container
        inline const std::vector<Node<OT>>& GetSolutions() const { return solutions; }

        // Get the ith solution in this container
        inline const Node<OT>& Get(size_t ix) const { return solutions[ix]; }
		
        // Get a mutable reference to the solution at index ix 
        inline Node<OT>& GetMutable(size_t ix) { return solutions[ix]; }

        // Remove all solutions from the container with more than num_nodes nodes
		void FilterOnNumberOfNodes(int num_nodes) {
			solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
            [num_nodes, this](const Node<OT>& n) -> bool {
            return n.NumNodes() > num_nodes;
        }), solutions.end());
		}

        // Checks if this container dominates node
        // If the number of nodes should also be compared, use node_compare = true
		bool Dominates(const Node<OT>& node, bool node_compare = true) const {
			for (size_t i = 0; i < Size(); i++) {
				if (Container<OT>::Dominates(solutions[i], node, node_compare)) return true;
			}
			return false;
		}

        // Checks if this container reverse dominates node
        // If the number of nodes should also be compared, use node_compare = true
        bool DominatesInv(const Node<OT>& node, bool node_compare = true) const {
			for (size_t i = 0; i < Size(); i++) {
				if (Container<OT>::DominatesInv(solutions[i], node, node_compare)) return true;
			}
			return false;
		}
		
        // Checks if this container stritcly dominates node
		bool StrictDominates(const Node<OT>& node) const {
			for (size_t i = 0; i < Size(); i++) {
				if (Container<OT>::StrictDominates(solutions[i], node)) return true;
			}
			return false;
		}

        // Checks if this container stritcly reverse dominates node
        bool StrictDominatesInv(const Node<OT>& node) const {
			for (size_t i = 0; i < Size(); i++) {
				if (Container<OT>::StrictDominatesInv(solutions[i], node)) return true;
			}
			return false;
		}

        // Checks if node n1 dominates node n2
        // If the number of nodes should also be compared, use node_compare = true
		static bool Dominates(const Node<OT>& n1, const Node<OT>& n2, bool node_compare = true) {
			return OT::Dominates(n1.solution, n2.solution)
				&& (!node_compare
					|| !(n1.solution == n2.solution)
					|| n1.NumNodes() <= n2.NumNodes());
		}

        // Checks if node n1 reverse dominates node n2
        // If the number of nodes should also be compared, use node_compare = true
        static bool DominatesInv(const Node<OT>& n1, const Node<OT>& n2, bool node_compare = true) {
			return OT::DominatesInv(n1.solution, n2.solution)
				&& (!node_compare
					|| !(n1.solution == n2.solution)
					|| n1.NumNodes() <= n2.NumNodes());
		}
		
        // Checks if node n1 strictly dominates node n2
		inline static bool StrictDominates(const Node<OT>& n1, const Node<OT>& n2) {
			return n1.solution != n2.solution && OT::Dominates(n1.solution, n2.solution);
		}

        // Checks if node n1 strictly reverse dominates node n2
        inline static bool StrictDominatesInv(const Node<OT>& n1, const Node<OT>& n2) {
			return n1.solution != n2.solution && OT::DominatesInv(n1.solution, n2.solution);
		}

        // remove temporary data from the container (such as the unique map)
        // This is useful to reduce memory usage and copy time when storing a container in the cache
		inline void RemoveTempData() { uniques.clear(); }

        // Set the size budget that was used to obtain this solution container
        // currently not used
		inline void SetSizeBudget(int max_depth, int max_num_nodes) { this->max_depth = max_depth; this->max_num_nodes = max_num_nodes; }

	private:

        // Add a node to the container
		template <bool inv>
		void InternalAdd(const Node<OT>& node) {
            
            // Add the node if the container is empty
            if (Size() == 0) {
                solutions.push_back(node);
                if constexpr (OT::check_unique) uniques[node.solution] = node.NumNodes();
                return;
            }

            // Test if the solution is already in the unique map
            // If so, compare the number of nodes and return if it is worse 
            if constexpr (OT::check_unique) {
                auto it = uniques.find(node.solution);
                if (it != uniques.end() && it->second <= node.NumNodes()) return;
                else if (it != uniques.end()) it->second = node.NumNodes();
                else uniques[node.solution] = node.NumNodes();
            }

            if constexpr (true) {
                // Only add it if the node is not dominated by this container
                for (size_t i = 0; i < Size(); i++) {
                    if constexpr(inv) {
                        if (Container<OT>::DominatesInv(solutions[i].solution, node.solution, 0)) return;
                    } else {
                        if (Container<OT>::Dominates(solutions[i].solution, node.solution, 0)) return;
                    }
                }

                // Remove dominated solutions from the container
                size_t old_size = solutions.size();
                if constexpr(inv) {
                    solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                        [&node, this](const Node<OT>& n) -> bool {
                        return Container<OT>::DominatesInv(node.solution, n.solution);
                    }), solutions.end());
                } else {
                    solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                        [&node, this](const Node<OT>& n) -> bool {
                        return Container<OT>::Dominates(node.solution, n.solution);
                    }), solutions.end());
                }

                // Add this solution to the container
                solutions.push_back(node);
            } else {
                // Add the node using binary search, to result in a sorted container.
                // Currently not used. Might be useful for some optimization tasks? ]
                ssize_t l = 0;
                ssize_t u = Size() - 1;
                ssize_t m;

                while (l <= u) {
                    m = (l + u) / 2;
                    if (node.solution == solutions[m].solution) return;
                    if (OT::FrontLT(node.solution, solutions[m].solution)) {
                        u = m - 1;
                    } else {
                        l = m + 1;
                    }
                }

                /* at the end of loop u is position just less than p, l is next position and > p */
                /* if the new point is dominated dont add */
                if (u >= 0 && Container<OT>::Dominates<inv>(solutions[u].solution, node.solution, 0)) return;

                /* Move past dominated points */
                for (; l < Size() && Container<OT>::Dominates<inv>(node.solution, solutions[l].solution, 0); l++);

                /* no points dominated if u+1 == l */
                if (u + 1 == l) {
                    solutions.insert(solutions.begin() + l, node);
                    return;
                }
                /* 1 point dominated if u+2 == l, simply replace */
                if (u + 2 == l) {
                    solutions[u + 1] = node;
                    return;
                }
                solutions[u + 1] = node;
                for (ssize_t i = u + 2; i < Size() && i < (Size() + u - l + 2); i++)
                    solutions[i] = solutions[i + l - u - 2];
                solutions.resize(Size() + u - l + 2);// ->length += u - l + 2;        
            }
        
	    }

		template <bool inv>
		void InternalAddD0(OT* task, const Node<OT>& node) {
			if (Size() == 0) {
				solutions.push_back(node);
				uniques[node.solution] = node.NumNodes();
				return;
			}

			//Test Unique
			auto it = uniques.find(node.solution);
			if (it != uniques.end() && it->second <= node.NumNodes()) return;
			else if (it != uniques.end()) it->second = node.NumNodes();
			else uniques[node.solution] = node.NumNodes();

			for (size_t i = 0; i < Size(); i++) {
                if constexpr(inv) {
                    if (task->DominatesD0Inv(solutions[i].solution, node.solution)) return;
                } else {
                    if (task->DominatesD0(solutions[i].solution, node.solution)) return;
                }
			}

			size_t old_size = solutions.size();
			if constexpr(inv) {
                solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                    [&node, &task, this](const Node<OT>& n) -> bool {
                    return task->DominatesD0Inv(node.solution, n.solution);
                }), solutions.end());
            } else {
                solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                    [&node, &task, this](const Node<OT>& n) -> bool {
                    return task->DominatesD0(node.solution, n.solution);
                }), solutions.end());
            }

			solutions.push_back(node);
		}

		template <bool inv, bool merge_inv>
		void InternalAddOrMerge(const Node<OT>& node, size_t max_size) {
        if (Size() == 0) {
            solutions.push_back(node);
            uniques[node.solution] = node.NumNodes();
            return;
        }

        //Test Unique
        auto it = uniques.find(node.solution);
        if (it != uniques.end() && it->second <= node.NumNodes()) return;
        else if (it != uniques.end()) it->second = node.NumNodes();
        else uniques[node.solution] = node.NumNodes();


        for (size_t i = 0; i < Size(); i++) {
            if constexpr(inv) {
                if (Container<OT>::DominatesInv(solutions[i].solution, node.solution, 0)) return;
            } else {
                if (Container<OT>::Dominates(solutions[i].solution, node.solution, 0)) return;
            }
            
        }

        size_t old_size = solutions.size();
        if constexpr(inv) {
            solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                [&node](const Node<OT>& n) -> bool {
                return Container<OT>::DominatesInv(node.solution, n.solution);
            }), solutions.end());
        } else {
            solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                [&node](const Node<OT>& n) -> bool {
                return Container<OT>::Dominates(node.solution, n.solution);
        }), solutions.end());
        }
        

        if (solutions.size() < max_size) {
            solutions.push_back(node);
        } else {
            size_t most_similar = 0;
            double min_diff = DBL_MAX;
            double diff;
            for (size_t i = 0; i < Size(); i++) {
                diff = OT::CalcDiff(solutions[i].solution, node.solution);
                if (diff < min_diff) {
                    most_similar = i;
                    min_diff = diff;
                }
            }
            auto& sim_node = solutions[most_similar];
            if constexpr(merge_inv) {
                OT::MergeInv(node.solution, sim_node.solution, sim_node.solution);
            } else {
                OT::Merge(node.solution, sim_node.solution, sim_node.solution);
            }
            
            
            if constexpr(inv) {
                solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                    [&node, this](const Node<OT>& n) -> bool {
                    return Container<OT>::StrictDominatesInv(node.solution, n.solution);
            }), solutions.end());
            } else {
                solutions.erase(std::remove_if(solutions.begin(), solutions.end(),
                    [&node, this](const Node<OT>& n) -> bool {
                    return Container<OT>::StrictDominates(node.solution, n.solution);
            }), solutions.end());
            }
            
        }
    }

        // The list of all solutions
		std::vector<Node<OT>> solutions;
        
        // A map of solutions to node count. For quick checking if an equivalent solution is already
        // part of this container, and how many nodes were used for this solution
        // Only relevant if OT::check_unique is true
		std::unordered_map<typename OT::SolType, int> uniques; 
		
        int max_num_nodes{ 0 }; // This value is currently not used
		int max_depth{ 0 };     // This value is currently not used
	};

}
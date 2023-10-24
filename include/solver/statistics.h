#pragma once

namespace STreeD {

	struct Statistics {
		Statistics() {
			num_terminal_nodes_with_node_budget_one = 0;
			num_terminal_nodes_with_node_budget_two = 0;
			num_terminal_nodes_with_node_budget_three = 0;

			total_time = 0;
			time_in_terminal_node = 0;
			time_merging = 0;
			time_lb_merging = 0;
			time_ub_subtracting = 0;
			time_reconstructing = 0;

			num_cache_hit_nonzero_bound = 0;
			num_cache_hit_optimality = 0;
		}

		void Print() {
			std::cout << "Total time elapsed: " << total_time << std::endl;
			std::cout << "\tTerminal time: " << time_in_terminal_node << std::endl;
			std::cout << "\tMerging time: " << time_merging << std::endl;
			std::cout << "\tLB Merging time: " << time_lb_merging << std::endl;
			std::cout << "\tUB Substracting time: " << time_ub_subtracting << std::endl;
			std::cout << "\tReconstructing time: " << time_reconstructing << std::endl;
			
			std::cout << "Terminal calls: " << num_terminal_nodes_with_node_budget_one + num_terminal_nodes_with_node_budget_two + num_terminal_nodes_with_node_budget_three << std::endl;
			std::cout << "\tTerminal 1 node: " << num_terminal_nodes_with_node_budget_one << std::endl;
			std::cout << "\tTerminal 2 node: " << num_terminal_nodes_with_node_budget_two << std::endl;
			std::cout << "\tTerminal 3 node: " << num_terminal_nodes_with_node_budget_three << std::endl;
		}

		size_t num_terminal_nodes_with_node_budget_one;
		size_t num_terminal_nodes_with_node_budget_two;
		size_t num_terminal_nodes_with_node_budget_three;

		size_t num_cache_hit_optimality;
		size_t num_cache_hit_nonzero_bound;


		double total_time;
		double time_in_terminal_node;
		double time_lb_merging;
		double time_ub_subtracting;
		double time_merging;
		double time_reconstructing;
	};
}
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from classes.dialogue_turn import DialogueTurn
from classes.result_json import Result
from classes.node_result import NodeResult

node_results = NodeResult.load_from_csv('/home/hasegawa_tomokazu/create_tree/result_json/baseline/ver.1.1-20.txt/NodeResult.csv')

# Extract the distance_from_ans values and filter out values that are not between 6 and 25
distances_ans = [node.distance_from_ans for node in node_results if 1 <= node.distance_from_ans <= 25]

# Extract the absolute differences and filter out values that are not between 6 and 25
distance_gpt = [abs(node.current_node_id_index - node.gpt_ans_node_id_index) for node in node_results if 1 <= abs(node.current_node_id_index - node.gpt_ans_node_id_index) <= 25]

# Plot the distribution
sns.histplot(distances_ans, kde=True, color='blue', label='Distance from Answer')
sns.histplot(distance_gpt, kde=True, color='red', label='Distance from GPT Answer')

plt.title('Distribution of Distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend()

# Save the plot as a PNG file
plt.savefig('/home/hasegawa_tomokazu/create_tree/plots/distance_distribution.png')

# Show the plot
plt.show()

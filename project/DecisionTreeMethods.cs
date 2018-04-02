using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.DecisionTree.Helpers;
using SimuKit.ML.Lang;

namespace SimuKit.ML.DecisionTree
{
    public class DecisionTreeMethods
    {
        public delegate double DoAttributeCostGainHandle(double gain, string feature_name);

        public static void Split<T>(DecisionTreePartition<T> partition, HashSet<string> feature_variable_names, DoAttributeCostGainHandle handle_cost)
            where T : DDataRecord
        {
            if (feature_variable_names.Count == 0)
            {
                Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_class_variable = SplitPartitionByClassVariable(partition, handle_cost);
                partition.UpdateSubPartitions(sub_partitions_by_class_variable, DecisionTree<T>.ClassVariableName);
                return;
            }

            double max_information_gain = 0;
            string selected_feature_name = null;

            HashSet<string> temp_feature_variable_names = null;
            if (partition.Forest == null)
            {
                temp_feature_variable_names = feature_variable_names.Clone();
            }
            else // if in random forest, select a random subset of features to split
            {
                int subset_featire_variable_count = partition.Forest.FeatureSubsetSize;
                List<string> temp_names = feature_variable_names.ToList();
                temp_names.Shuffle();

                temp_feature_variable_names = new HashSet<string>();
                for (int i = 0; i < subset_featire_variable_count; ++i)
                {
                    temp_feature_variable_names.Add(temp_names[i]);
                }
            }

            foreach (string variable_name in temp_feature_variable_names)
            {
                double g = CalcInformationGain(partition, variable_name, handle_cost);
                if (max_information_gain < g)
                {
                    max_information_gain = g;
                    selected_feature_name = variable_name;
                }
            }

            partition.SplitInformationGain = max_information_gain;

            if (selected_feature_name == null) // this suggest the children of the nodes are leave nodes
            {
                Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_class_variable = SplitPartitionByClassVariable(partition, handle_cost);
                partition.UpdateSubPartitions(sub_partitions_by_class_variable, DecisionTree<T>.ClassVariableName);
            }
            else
            {
                if (partition.Forest == null) // if not in random forest, remove the used feature variable; otherwise, keep all and in subsequent split, select random subset
                {
                    feature_variable_names.Remove(selected_feature_name);
                }

                Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_feature_name = SplitPartitionByFeatureVariable(partition, selected_feature_name, handle_cost);
                partition.UpdateSubPartitions(sub_partitions_by_feature_name, selected_feature_name);

                if (feature_variable_names.Count > 0)
                {
                    foreach (DecisionTreePartition<T> sub_partition_by_feature_name in sub_partitions_by_feature_name.Values)
                    {
                        Split(sub_partition_by_feature_name, feature_variable_names.Clone(), handle_cost);
                    }
                }
            }
        }

        public static DecisionTreeNode<T> CreateRootNode<T>(DecisionTree<T> tree)
            where T: DDataRecord
        {
            return new DecisionTreeNode<T>(tree, null, null);
        }

        public static DecisionTreeNode<T> CreateTreeNode<T>(string variable_name, string variable_value, DecisionTreeNode<T> tree_node)
            where T : DDataRecord
        {
            return new DecisionTreeNode<T>(tree_node.Tree, variable_name, variable_value, tree_node);
        }

        public static DecisionTreeNode<T> CreateLeaveNode<T>(string variable_name, string variable_value, DecisionTreeNode<T> tree_node, int record_count)
            where T : DDataRecord
        {
            return new DecisionTreeNode<T>(tree_node.Tree, variable_name, variable_value, tree_node, record_count);
        }

        public static Dictionary<string, DecisionTreePartition<T>> SplitPartitionByFeatureVariable<T>(DecisionTreePartition<T> partition, string variable_name, DoAttributeCostGainHandle handle_cost)
             where T : DDataRecord
        {
            IEnumerable<T> data_store = partition.DataStore;
            DecisionTreeNode<T> tree_node = partition.TreeNode;

            Dictionary<string, DecisionTreePartition<T>> sub_partitions = new Dictionary<string, DecisionTreePartition<T>>();

            Dictionary<string, List<T>> sub_data_stores = new Dictionary<string, List<T>>();
            foreach (T rec in data_store)
            {
                string variable_value = rec[variable_name];
                List<T> sub_data_store = null;

                if (sub_data_stores.ContainsKey(variable_value))
                {
                    sub_data_store = sub_data_stores[variable_value];
                }
                else
                {
                    sub_data_store = new List<T>();
                    sub_data_stores[variable_value] = sub_data_store;
                }

                sub_data_store.Add(rec);
            }

            foreach (string variable_value in sub_data_stores.Keys)
            {
                List<T> sub_data_store = sub_data_stores[variable_value];

                DecisionTreeNode<T> sub_node = DecisionTreeMethods.CreateTreeNode<T>(variable_name, variable_value, tree_node);
                sub_node.HandleCost += (gain, fname) =>
                {
                    if (handle_cost != null)
                    {
                        return handle_cost(gain, fname);
                    }
                    return gain;
                };

                sub_partitions[variable_value] = DecisionTreePartition<T>.Create(sub_node, sub_data_store);
            }

            return sub_partitions;
        }

        public static double CalcEntropy<T>(DecisionTreePartition<T> partition, Dictionary<string, DecisionTreePartition<T>> sub_partitions)
             where T : DDataRecord
        {
            double prob = 0;
            double entropy = 0;

            foreach (string variable_value in sub_partitions.Keys)
            {
                prob = (double)sub_partitions[variable_value].RecordCount / partition.RecordCount;
                entropy += (-prob * System.Math.Log(prob) / System.Math.Log(2));
            }

            return entropy;
        }

        public static Dictionary<string, DecisionTreePartition<T>> SplitPartitionByClassVariable<T>(DecisionTreePartition<T> partition, DoAttributeCostGainHandle handle_cost)
             where T : DDataRecord
        {
            IEnumerable<T> data_store = partition.DataStore;
            DecisionTreeNode<T> tree_node = partition.TreeNode;

            IEnumerable<string> class_variable_value_iterator = tree_node.ClassVariableValueIterator;

            Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_class_variable = new Dictionary<string, DecisionTreePartition<T>>();

            foreach (string variable_value in class_variable_value_iterator)
            {
                int record_count = tree_node.FindClassVariableValueCount(variable_value);
                DecisionTreeNode<T> sub_node = CreateLeaveNode(DecisionTree<T>.ClassVariableName, variable_value, tree_node, record_count);

                sub_node.HandleCost += (gain, fname) =>
                {
                    if (handle_cost != null)
                    {
                        return handle_cost(gain, fname);
                    }
                    return gain;
                };
                sub_partitions_by_class_variable[variable_value] = DecisionTreePartition<T>.Create(sub_node);
            }

            return sub_partitions_by_class_variable;
        }

        public static double CalcInformationGain<T>(DecisionTreePartition<T> partition, string feature_variable_name, DoAttributeCostGainHandle handle_cost)
             where T : DDataRecord
        {
            Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_feature_variable = SplitPartitionByFeatureVariable(partition, feature_variable_name, handle_cost);

            Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_class_variable = SplitPartitionByClassVariable(partition, handle_cost);

            double gain = CalcEntropy(partition, sub_partitions_by_class_variable);

            foreach (string variable_value in sub_partitions_by_feature_variable.Keys)
            {
                DecisionTreePartition<T> sub_partition_by_feature = sub_partitions_by_feature_variable[variable_value];
                Dictionary<string, DecisionTreePartition<T>> sub_sub_partitions = SplitPartitionByClassVariable(sub_partition_by_feature, handle_cost);
                double entropy = CalcEntropy(sub_partition_by_feature, sub_sub_partitions);
                gain -= (sub_partition_by_feature.RecordCount * entropy / partition.RecordCount);
            }

            if (handle_cost != null)
            {
                gain = handle_cost(gain, feature_variable_name);
            }


            return gain;
        }

    }
}

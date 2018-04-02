using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Lang;

namespace SimuKit.ML.DecisionTree
{
    public class DecisionTreePartition<T>
         where T : DDataRecord
    {
        protected IEnumerable<T> mDataStore;
        protected DecisionTreeNode<T> mTreeNode;
        protected string mSplitVariableName;

        public static DecisionTreePartition<T> Create(DecisionTreeNode<T> tree_node, IEnumerable<T> data_store)
        {
            foreach (T rec in data_store)
            {
                tree_node.Scan(rec);
            }
            return new DecisionTreePartition<T>(tree_node, data_store);
        }

        public static DecisionTreePartition<T> Create(DecisionTreeNode<T> tree_node)
        {
            return new DecisionTreePartition<T>(tree_node, null);
        }

        public RandomForest<T> Forest
        {
            get
            {
                return mTreeNode.Tree.Forest;
            }
        }

        private DecisionTreePartition(DecisionTreeNode<T> tree_node, IEnumerable<T> data_store)
        {
            mTreeNode = tree_node;
            mDataStore = data_store;
        }

        public string SplitVariableName
        {
            get { return mTreeNode.SplitVariableName; }
            set { mTreeNode.SplitVariableName = value; }
        }

        public double SplitInformationGain
        {
            get { return mTreeNode.InformationGain; }
            set { mTreeNode.InformationGain = value; }
        }

        public int RecordCount
        {
            get
            {
                return mTreeNode.RecordCount;
            }
        }

        public IEnumerable<T> DataStore
        {
            get { return mDataStore; }
        }

        public DecisionTreeNode<T> TreeNode
        {
            get { return mTreeNode; }
        }

        public string VariableName
        {
            get { return mTreeNode.VariableName; }
        }

        public string VariableValue
        {
            get { return mTreeNode.VariableValue; }
        }

        public void UpdateSubPartitions(Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_variable, string split_variable_name)
        {
            SplitVariableName = split_variable_name;

            mTreeNode.RemoveAllChildren();
            foreach (string variable_value in sub_partitions_by_variable.Keys)
            {
                mTreeNode[variable_value] = sub_partitions_by_variable[variable_value].TreeNode;
            }
        }
    }
}

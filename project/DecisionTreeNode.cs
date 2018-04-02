using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Lang;
using System.Xml;
using SimuKit.ML.DecisionTree.Helpers;

namespace SimuKit.ML.DecisionTree
{
    public class DecisionTreeNode<T>
        where T : DDataRecord
    {
	    protected string mSplitVariableName = ""; //the feature name associated with the children of the node
        protected string mVariableName = ""; 
        protected string mVariableValue = ""; //the feature value at the node
	    
        protected Dictionary<string, DecisionTreeNode<T>> mChildren = new Dictionary<string,DecisionTreeNode<T>>(); //children for which the key is the feature value at each child
	    
        protected double mInformationGain=0;
        
        protected DecisionTreeNode<T> mParent = null;
        
        protected Dictionary<string, int> mClassVariableValueDistribution = new Dictionary<string, int>();

        protected int mRecordCount = 0;

        protected DecisionTree<T> mTree;

        public DecisionTree<T> Tree
        {
            get { return mTree; }
        }

        public delegate double DoAttributeCostGainHandle(double gain, string feature_name);
        public event DoAttributeCostGainHandle HandleCost;

        public IEnumerable<string> ClassVariableValueIterator
        {
            get
            {
                return mClassVariableValueDistribution.Keys;
            }
        }

        public string SplitVariableName
        {
            get { return mSplitVariableName; }
            set { mSplitVariableName = value; }
        }

        public string VariableName
        {
            get { return mVariableName; }
        }

        public int FindClassVariableValueCount(string class_variable_value)
        {
            if (mClassVariableValueDistribution.ContainsKey(class_variable_value))
            {
                return mClassVariableValueDistribution[class_variable_value];
            }
            return 0;
        }

        public void Scan(T rec)
        {
            mRecordCount++;
            string class_variable_value = rec.Label;
            if (mClassVariableValueDistribution.ContainsKey(class_variable_value))
            {
                mClassVariableValueDistribution[class_variable_value]++;
            }
            else
            {
                mClassVariableValueDistribution[class_variable_value] = 1;
            }
        }

        public string VariableValue
        {
            get { return mVariableValue; }
        }
	
	    public void RemoveAllChildren()
	    {
            mChildren.Clear();
	    }

        public DecisionTreeNode<T> this[string variable_value]
        {
            get
            {
                return mChildren[variable_value];
            }
            set
            {
                mChildren[variable_value] = value;
            }
        }

        public DecisionTreeNode<T> Parent
        {
            get { return mParent; }
        }
	
	    public DecisionTreeNode(DecisionTree<T> tree, string variable_name, string variable_value, DecisionTreeNode<T> parent=null, int record_count = 0)
	    {
            mTree = tree;
            mParent = parent;
            mVariableName = variable_name;
            mVariableValue = variable_value;
            mRecordCount = record_count;
	    }
	
	    public int RecordCount
	    {
            get
            {
                return mRecordCount;
            }
	    }

        public void FlattenBranchNodes(List<DecisionTreeNode<T>> nodes)
        {
            if (mSplitVariableName != DecisionTree<T>.ClassVariableName)
            {
                nodes.Add(this);
                foreach (DecisionTreeNode<T> child in mChildren.Values)
                {
                    child.FlattenBranchNodes(nodes);
                }
            }
        }

        public void FlattenLeaveNodes(List<DecisionTreeNode<T>> nodes)
        {
            if (mSplitVariableName == DecisionTree<T>.ClassVariableName)
            {
                nodes.Add(this);
            }
            else
            {
                foreach (DecisionTreeNode<T> child in mChildren.Values)
                {
                    child.FlattenLeaveNodes(nodes);
                }
            }
        }
	
	    public double InformationGain
	    {
            get
            {
                return mInformationGain;
            }
            set
            {
                mInformationGain = value;
            }
	    }

        public string PredictedLabel
        {
            get
            {
                if (mSplitVariableName == DecisionTree<T>.ClassVariableName)
                {
                    string predicted_label = null;
                    int max_count = 0;
                    foreach (string class_variable_value in mChildren.Keys)
                    {
                        int rscount = mChildren[class_variable_value].RecordCount;
                        if (rscount > max_count)
                        {
                            max_count = rscount;
                            predicted_label = class_variable_value;
                        }
                    }

                    return predicted_label;
                }
                else
                {
                    throw new ArgumentNullException();
                }
            }
        }

        public string Predict(T record)
	    {
		    if(mSplitVariableName == DecisionTree.ClassVariableName)
		    {
                return PredictedLabel;
		    }
		    else 
		    {
			    string featureValue = record[mSplitVariableName];

                if (record.IsCategorical(mSplitVariableName))
                {
                    if (mChildren.ContainsKey(featureValue))
                    {
                        DecisionTreeNode child = mChildren[featureValue];
                        return child.Predict(record);
                    }
                    else
                    {
                        return null;
                    }
                } else
                {
                    double numericFeatureValue = 0;
                    if(double.TryParse(featureValue, out numericFeatureValue))
                    {
                        foreach(string key in mChildren.Keys)
                        {
                            if (key.StartsWith("<="))
                            {
                                double cutoff = double.Parse(key.Replace("<=", "").Trim());
                                if(numericFeatureValue <= cutoff)
                                {
                                    return mChildren[key].Predict(record);
                                }
                            } else
                            {
                                double cutoff = double.Parse(key.Replace(">", "").Trim());
                                if(numericFeatureValue > cutoff)
                                {
                                    return mChildren[key].Predict(record);
                                }
                            }
                        }
                    } else
                    {
                        List<string> keys = mChildren.Keys.ToList();
                        string selectedKey = keys[RandomEngine.NextInt(keys.Count)];
                        return mChildren[selectedKey].Predict(record);
                    }
                }

			    
		    }

            return null;
	    }

        public KeyValuePair<string, Dictionary<string, DecisionTreeNode<T>>> Prune()
        {
            Dictionary<string, DecisionTreeNode<T>> children = mChildren;
            KeyValuePair<string, Dictionary<string, DecisionTreeNode<T>>> tree_info = new KeyValuePair<string, Dictionary<string, DecisionTreeNode<T>>>(
                mSplitVariableName,
                children);

            DecisionTreePartition<T> partition = DecisionTreePartition<T>.Create(this);
            Dictionary<string, DecisionTreePartition<T>> sub_partitions_by_class_variable = DecisionTreeMethods.SplitPartitionByClassVariable<T>(partition, (gain, feature_name) =>
                {
                    if (HandleCost != null)
                    {
                        return HandleCost(gain, feature_name);
                    }
                    return gain;
                });

            partition.UpdateSubPartitions(sub_partitions_by_class_variable, DecisionTree<T>.ClassVariableName);

            return tree_info;
        }

        public void Join(KeyValuePair<string, Dictionary<string, DecisionTreeNode<T>>> tree_info)
        {
            mSplitVariableName = tree_info.Key;
            mChildren = tree_info.Value;
        }

        /// <summary>
        /// split for the root node
        /// </summary>
        public void Split(HashSet<string> feature_names, IEnumerable<T> data_store)
        {
            DecisionTreePartition<T> root_partition = DecisionTreePartition<T>.Create(this, data_store);
            DecisionTreeMethods.Split<T>(root_partition, feature_names, (gain, feature_name) =>
                {
                    if (HandleCost != null)
                    {
                        return HandleCost(gain, feature_name);
                    }
                    return gain;
                });
        }

        public virtual XmlElement CreateXml(XmlDocument doc)
        {
            XmlElement node_element = doc.CreateElement("DecisionTreeNode");
            node_element.AppendAttribute("SplitVariableName", mSplitVariableName);
            node_element.AppendAttribute("VariableName", mVariableName);
            node_element.AppendAttribute("VariableValue", mVariableValue);
            node_element.AppendAttribute("InformationGain", mInformationGain);
            node_element.AppendAttribute("RecordCount", mRecordCount);

            XmlElement class_variable_value_distribution_element = doc.CreateElement("ClassVariableValueDistribution");
            node_element.AppendChild(class_variable_value_distribution_element);
            foreach (string class_variable_value in mClassVariableValueDistribution.Keys)
            {
                XmlElement class_variable_value_element = doc.CreateElement("ClassVariableValue");
                class_variable_value_distribution_element.AppendAttribute("RecordCount", mClassVariableValueDistribution[class_variable_value]);
                class_variable_value_distribution_element.AppendAttribute("VariableValue", class_variable_value);
                class_variable_value_distribution_element.AppendChild(class_variable_value_element);
            }

            XmlElement child_nodes_element = doc.CreateElement("ChildNodes");
            node_element.AppendChild(child_nodes_element);
            foreach (string split_variable_value in mChildren.Keys)
            {
                DecisionTreeNode<T> child_node = mChildren[split_variable_value];
                XmlElement child_node_element = child_node.CreateXml(doc);
                child_nodes_element.AppendChild(child_node_element);
            }

            return node_element;
        }


        public void BuildRule(Rule<T> rule)
        {
            if (mParent != null)
            {
                string feature_name = mParent.mSplitVariableName;
                rule.Antecedents[feature_name] = mVariableValue;
            }

            if (mSplitVariableName == DecisionTree<T>.ClassVariableName)
            {    
                rule.Consequent = new KeyValuePair<string, string>("Predicted Label", PredictedLabel);
            }
        }

        public void InitializeFromXml(XmlElement node_element)
        {
            node_element.TryQueryStringAttribute("SplitVariableName", out mSplitVariableName);

            node_element.TryQueryStringAttribute("VariableName", out mVariableName);
            node_element.TryQueryStringAttribute("VariableValue", out mVariableValue);
            node_element.TryQueryDoubleAttribute("InformationGain", out mInformationGain);
            node_element.TryQueryIntAttribute("RecordCount", out mRecordCount);

            foreach (XmlElement nodemChildren_element in node_element.ChildNodes)
            {
                if (nodemChildren_element.Name == "ClassVariableValueDistribution")
                {
                    foreach (XmlElement class_variable_value_node in nodemChildren_element.ChildNodes)
                    {
                        if (class_variable_value_node.Name == "ClassVariableValue")
                        {
                            int variable_value_count = 0;
                            string variable_value_name = "";
                            class_variable_value_node.TryQueryStringAttribute("VariableValue", out variable_value_name);
                            class_variable_value_node.TryQueryIntAttribute("RecordCount", out variable_value_count);
                            mClassVariableValueDistribution[variable_value_name] = variable_value_count;
                        }
                    }
                }
                else if (nodemChildren_element.Name == "ChildNodes")
                {
                    foreach (XmlElement child_node_element in nodemChildren_element.ChildNodes)
                    {
                        DecisionTreeNode<T> child_node = new DecisionTreeNode<T>(mTree, null, null, this);
                        child_node.InitializeFromXml(child_node_element);
                        mChildren[child_node.VariableValue] = child_node;
                    }
                }
            }
        }
    }
}

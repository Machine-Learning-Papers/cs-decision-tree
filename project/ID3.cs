using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Lang;
using SimuKit.ML.Solvers;
using System.Xml;

namespace SimuKit.ML.DecisionTree
{
    public class ID3<T> : Classifier<T, string>
        where  T : DDataRecord
    {
        protected DecisionTree<T> mTree;
        protected bool mEnableErrorReducePruning = false;

        public delegate double DoAttributeCostGainHandle(double gain, string feature_name);
        public event DoAttributeCostGainHandle DoAttributeCostGain;

        public RandomForest<T> Forest
        {
            get { return mTree.Forest; }
            set { mTree.Forest = value; }
        }

        public bool EnableErrorReducePruning
        {
            get { return mEnableErrorReducePruning; }
            set { mEnableErrorReducePruning = value; }
        }

        public ID3()
        {
            mTree = new DecisionTree<T>();
            mTree.HandleCost += (gain, feature_name) =>
            {
                if (DoAttributeCostGain != null)
                {
                    return DoAttributeCostGain(gain, feature_name);
                }
                return gain;
            };
        }

        public DecisionTree<T> Model
        {
            get { return mTree; }
        }

        public void RandomShuffle(List<T> sample)
        {
            Random rng = new Random();
            int n = sample.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = sample[k];
                sample[k] = sample[n];
                sample[n] = value;
            }
        }

        public XmlDocument CreateXml()
        {
            return mTree.CreateXml();
        }

        public void WriteToXml(string filepath)
        {
            mTree.WriteToXml(filepath);
        }

        public void InitializeFromXml(XmlDocument doc)
        {
            mTree.InitializeFromXml(doc);
        }

        public void ReadFromXml(string filepath)
        {
            mTree.ReadFromXml(filepath);
        }

        public override void Train(List<T> records)
        {
            mTree.Clear();

            string[] feature_names = records[0].FindFeatures();
            HashSet<string> feature_name_set = new HashSet<string>();
            foreach (string feature_name in feature_names)
            {
                feature_name_set.Add(feature_name);
            }

            mTree.Build(feature_name_set, records);
        }

        public override string Predict(T record)
        {
            return mTree.Predict(record);
        }

        public override void Predict(List<T> data_set)
        {
            foreach (T rec in data_set)
            {
                rec.PredictedLabel = Predict(rec);
            }
        }

        public override double ComputeCost(List<T> data_set)
        {
            throw new NotImplementedException();
        }

        public void UpdateContinuousAttributes(List<T> sample, string feature_name)
        {
            List<KeyValuePair<double, string>> ordered_values = new List<KeyValuePair<double, string>>();
            foreach (T rec in sample)
            {
                string value_string = rec[feature_name];
                double value;
                double.TryParse(value_string, out value);
                ordered_values.Add(new KeyValuePair<double, string>(value, rec.Label));
            }

            ordered_values.Sort((kvp1, kvp2) =>
            {
                return kvp1.Key.CompareTo(kvp2.Key);
            });

            List<double> boundaries = new List<double>();
            for (int i = 0; i < ordered_values.Count - 1; ++i)
            {
                int j = i + 1;
                double value1 = ordered_values[i].Key;
                double value2 = ordered_values[j].Key;
                string class_variable_value1 = ordered_values[i].Value;
                string class_variable_value2 = ordered_values[j].Value;
                if (class_variable_value1 != class_variable_value2)
                {
                    boundaries.Add((value1 + value2) / 2);
                }
            }

            double max_information_gain = double.MinValue;
            double best_boundary_value = 0;
            foreach (double boundary_value in boundaries)
            {
                int count_lower = 0;
                for (int i = 0; i < ordered_values.Count; ++i)
                {
                    if (ordered_values[i].Key <= boundary_value)
                    {
                        count_lower++;
                    }
                    else
                    {
                        break;
                    }
                }
                int count_higher = ordered_values.Count - count_lower;

                Dictionary<string, int> partitions = new Dictionary<string, int>();
                for (int i = 0; i < count_lower; ++i)
                {
                    string class_variable_value = ordered_values[i].Value;
                    if (partitions.ContainsKey(class_variable_value))
                    {
                        partitions[class_variable_value]++;
                    }
                    else
                    {
                        partitions[class_variable_value] = 1;
                    }
                }
                double entropy_lower = 0;
                foreach (string class_variable_value in partitions.Keys)
                {
                    int partition_size = partitions[class_variable_value];
                    double p = (double)partition_size / count_lower;
                    entropy_lower += (-p * System.Math.Log(p, 2));
                }

                partitions = new Dictionary<string, int>();
                for (int i = count_lower; i < ordered_values.Count; ++i)
                {
                    string class_variable_value = ordered_values[i].Value;
                    if (partitions.ContainsKey(class_variable_value))
                    {
                        partitions[class_variable_value]++;
                    }
                    else
                    {
                        partitions[class_variable_value] = 1;
                    }
                }
                double entropy_higher = 0;
                foreach (string class_variable_value in partitions.Keys)
                {
                    int partition_size = partitions[class_variable_value];
                    double p = (double)partition_size / count_higher;
                    entropy_higher += (-p * System.Math.Log(p, 2));
                }


                partitions = new Dictionary<string, int>();
                for (int i = 0; i < ordered_values.Count; ++i)
                {
                    string class_variable_value = ordered_values[i].Value;
                    if (partitions.ContainsKey(class_variable_value))
                    {
                        partitions[class_variable_value]++;
                    }
                    else
                    {
                        partitions[class_variable_value] = 1;
                    }
                }
                double entropy_overral = 0;
                foreach (string class_variable_value in partitions.Keys)
                {
                    int partition_size = partitions[class_variable_value];
                    double p = (double)partition_size / count_higher;
                    entropy_overral += (-p * System.Math.Log(p, 2));
                }

                double information_gain = entropy_overral - (count_lower * entropy_lower / (ordered_values.Count) + count_higher * entropy_higher / (ordered_values.Count));
                if (information_gain > max_information_gain)
                {
                    max_information_gain = information_gain;
                    best_boundary_value = boundary_value;
                }
            }

            foreach (T rec in sample)
            {
                string value_string = rec[feature_name];
                double value;
                double.TryParse(value_string, out value);
                if (value <= best_boundary_value)
                {
                    rec[feature_name]=string.Format("<= {0}", best_boundary_value);
                }
                else
                {
                    rec[feature_name]=string.Format("> {0}", best_boundary_value);
                }
            }
        }

        public int Evaluate(List<T> Xval)
        {
            int error = 0;
            foreach (T rec in Xval)
            {
                if (Predict(rec) != rec.Label)
                {
                    error += 1;
                }
            }

            return error;
        }

        public int ErrorReducePrune(List<T> Xval, int original_error = -1)
        {
            if (original_error == -1)
            {
                original_error = Evaluate(Xval);
            }

            int max_improvement = 0;
            DecisionTreeNode<T> node_to_prune = null;
            List<DecisionTreeNode<T>> nodes = mTree.FlattenBranchNodes();
            int new_error = 0;
            foreach (DecisionTreeNode<T> node in nodes)
            {
                KeyValuePair<string, Dictionary<string, DecisionTreeNode<T>>> preserved_node_info = node.Prune();
                new_error = Evaluate(Xval);
                node.Join(preserved_node_info);
                int improvement = original_error - new_error;
                if (improvement > max_improvement)
                {
                    max_improvement = improvement;
                    node_to_prune = node;
                }
            }

            if (node_to_prune != null)
            {
                node_to_prune.Prune();
                return ErrorReducePrune(Xval, new_error);
            }

            return original_error;
        }
    }
}

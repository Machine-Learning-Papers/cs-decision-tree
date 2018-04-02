using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Lang;

namespace SimuKit.ML.DecisionTree
{
    using Lang;
    using System.Xml;
    using SimuKit.ML.Solvers;

    /// <summary>
    /// Decision Tree
    /// </summary>
    public class DecisionTree<T> 
        where T : DDataRecord
    {
        protected DecisionTreeNode<T> mRootNode;

        public delegate double DoAttributeCostGainHandle(double gain, string feature_name);
        public event DoAttributeCostGainHandle HandleCost;

        private RandomForest<T> mForest = null;

        public RandomForest<T> Forest
        {
            get { return mForest; }
            set { mForest = value; }
        }

        public const string ClassVariableName = "class_variable@decision_tree";

        public DecisionTree()
        {
            mRootNode = DecisionTreeMethods.CreateRootNode<T>(this);
            mRootNode.HandleCost += (gain, feature_name) =>
                {
                    if (HandleCost != null)
                    {
                        return HandleCost(gain, feature_name);
                    }
                    return gain;
                };
        }

        public List<Rule<T>> ToRules()
        {
            List<Rule<T>> rules = new List<Rule<T>>();
            List<DecisionTreeNode<T>> leaves = new List<DecisionTreeNode<T>>();
            mRootNode.FlattenLeaveNodes(leaves);
            foreach (DecisionTreeNode<T> leave in leaves)
            {
                Rule<T> rule = new Rule<T>();
                leave.BuildRule(rule);
                rules.Add(rule);
            }
            return rules;
        }

        public void Build(HashSet<string> feature_names, IEnumerable<T> data_store)
        {
            mRootNode.RemoveAllChildren();
            mRootNode.Split(feature_names, data_store);
        }

        public void Clear()
        {
            mRootNode = DecisionTreeMethods.CreateRootNode<T>(this);
            mRootNode.HandleCost += (gain, feature_name) =>
                {
                    if (HandleCost != null)
                    {
                        return HandleCost(gain, feature_name);
                    }
                    return gain;
                };
        }

        public List<DecisionTreeNode<T>> FlattenBranchNodes()
        {
            List<DecisionTreeNode<T>> nodes = new List<DecisionTreeNode<T>>();
            mRootNode.FlattenBranchNodes(nodes);
            return nodes;
        }

        public string Predict(T record)
        {
            return mRootNode.Predict(record);
        }

        public XmlDocument CreateXml()
        {
            XmlDocument doc = new XmlDocument();
            XmlElement xml_root = doc.CreateElement("DecisionTree");

            XmlElement root_node_element = mRootNode.CreateXml(doc);
            xml_root.AppendChild(root_node_element);

            doc.AppendChild(xml_root);
            return doc;
        }

        public void InitializeFromXml(XmlDocument doc)
        {
            XmlElement xml_root = doc.DocumentElement;
            foreach (XmlElement root_node_element in xml_root.ChildNodes)
            {
                mRootNode.InitializeFromXml(root_node_element);
            }
        }

        public void ReadFromXml(string filepath)
        {
            XmlDocument doc = new XmlDocument();
            doc.Load(filepath);
            InitializeFromXml(doc);
        }

        public void WriteToXml(string filepath)
        {
            XmlDocument doc=CreateXml();

            doc.Save(filepath);
        }
    }
}

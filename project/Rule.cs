using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Lang;

namespace DecisionTree
{
    public class Rule<T>
        where T : DDataRecord
    {
        protected Dictionary<string, string> mAntecedents = new Dictionary<string, string>();
        protected KeyValuePair<string, string> mConsequent = new KeyValuePair<string, string>();

        public Dictionary<string, string> Antecedents
        {
            get { return mAntecedents; }
        }

        public KeyValuePair<string, string> Consequent
        {
            get { return mConsequent; }
            set { mConsequent = value; }
        }

        public bool IsFired(T record)
        {
            foreach (string feature_name in mAntecedents.Keys)
            {
                string attribute_value = mAntecedents[feature_name];
                if (record[feature_name] != attribute_value)
                {
                    return false;
                }
            }
            return true;
        }
    }
}

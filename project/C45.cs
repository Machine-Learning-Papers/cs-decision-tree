using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Lang;
using System.Xml;

namespace SimuKit.ML.DecisionTree
{
    public class C45<T> : ID3<T>
        where T : DDataRecord
    {
        List<Rule<T>> mRules = null;
        Dictionary<string, double> mCosts=null;

        public C45()
        {

        }

        public List<Rule<T>> Rules
        {
            get { return mRules; }
        }

        public void EnableCostHandling_TanSchlimmer(Dictionary<string, double> costs)
        {
            mCosts = costs;
            mTree.HandleCost += (gain, feature_name) =>
                {
                    double gain_with_cost = gain * gain / (mCosts[feature_name]);
                    return gain_with_cost;
                };
        }

        public void EnableCostHandling_Nunez(Dictionary<string, double> costs, double w)
        {
            mCosts = costs;
            mTree.HandleCost += (gain, feature_name) =>
            {
                double gain_with_cost = (System.Math.Pow(2, gain) - 1) / System.Math.Pow(mCosts[feature_name] + 1, w);
                return gain_with_cost;
            };
        }

        public int RulePostPrune(List<T> Xval)
        {
            mRules = mTree.ToRules();

            List<KeyValuePair<Rule<T>, double>> estimated_rule_accuracies = new List<KeyValuePair<Rule<T>, double>>();

            // calculate the estimated rule accuracy for each rule 
            for (int i = 0; i < mRules.Count; ++i )
            {
                Rule<T> rule = mRules[i];
                int error_count = 0;
                foreach (T record in Xval)
                {
                    if (rule.IsFired(record))
                    {
                        if (rule.Consequent.Value != record.Label)
                        {
                            error_count++;
                        }
                    }
                    else
                    {
                        error_count++;
                    }
                }

                int n=Xval.Count;
                double observed_accuracy = (double)(n - error_count) / n;
                double binomial_stddev = System.Math.Sqrt(n * observed_accuracy * (1 - observed_accuracy));
                double estimated_rule_accuracy = (observed_accuracy - binomial_stddev * 1.96); //at 95% confidence level

                estimated_rule_accuracies.Add(new KeyValuePair<Rule<T>, double>(rule, estimated_rule_accuracy));
            }

            // for each rule, 
            for (int i = 0; i < mRules.Count; ++i)
            {
                Rule<T> rule = mRules[i];


                int iteration_count = rule.Antecedents.Count;
                for (int j = 0; j < iteration_count; ++ j )
                {
                    List<string> feature_names = rule.Antecedents.Keys.ToList();

                    double best_estimated_rule_accuracy = double.MinValue;
                    string pruned_candidate_attribute = null;
                    foreach (string feature_name in feature_names)
                    {
                        string attribute_value = rule.Antecedents[feature_name];
                        rule.Antecedents.Remove(feature_name);

                        int error_count = 0;
                        foreach (T record in Xval)
                        {
                            if (rule.IsFired(record))
                            {
                                if (rule.Consequent.Value != record.Label)
                                {
                                    error_count++;
                                }
                            }
                            else
                            {
                                error_count++;
                            }
                        }

                        int n = Xval.Count;
                        double observed_accuracy = (double)(n - error_count) / n;
                        double binomial_stddev = System.Math.Sqrt(n * observed_accuracy * (1 - observed_accuracy));
                        double estimated_rule_accuracy = (observed_accuracy - binomial_stddev * 1.96); //at 95% confidence level

                        rule.Antecedents[feature_name] = attribute_value;

                        if (best_estimated_rule_accuracy < estimated_rule_accuracy)
                        {
                            best_estimated_rule_accuracy = estimated_rule_accuracy;
                            pruned_candidate_attribute = feature_name;
                        }
                    }

                    if (best_estimated_rule_accuracy < estimated_rule_accuracies[i].Value) // only prune the rule when estimated rule accuracy increase!
                    {
                        break;
                    }
                    else
                    {
                        rule.Antecedents.Remove(pruned_candidate_attribute);
                        estimated_rule_accuracies[i] = new KeyValuePair<Rule<T>, double>(rule, best_estimated_rule_accuracy);
                    }
                }
            }

            estimated_rule_accuracies.Sort((KeyValuePair<Rule<T>, double> kvp1, KeyValuePair<Rule<T>, double> kvp2) =>
            {
                return kvp2.Value.CompareTo(kvp1.Value);
            });

            mRules.Clear();
            foreach (KeyValuePair<Rule<T>, double> kvp in estimated_rule_accuracies)
            {
                mRules.Add(kvp.Key);
            }

            return Evaluate(Xval);
        }

        public override string Predict(T record)
        {
            if (mRules == null || mRules.Count==0)
            {
                return base.Predict(record);
            }
            foreach (Rule<T> rule in mRules)
            {
                if (rule.IsFired(record))
                {
                    return rule.Consequent.Value;
                }
            }

            return null;
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


 

    }
}

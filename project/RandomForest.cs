using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SimuKit.ML.Solvers;
using SimuKit.ML.Lang;

namespace SimuKit.ML.DecisionTree
{
    /// <summary>
    /// IEnumerable&lt;DDataRecord&lt;string&gt;&gt; training_sample = LoadTrainingSamples();
    /// IEnumerable&lt;DDataRecord&lt;string&gt;&gt; testing_sample = LoadTestingSamples();
    /// 
    /// RandomForest&lt;DDataRecord&gt; classifier = new RandomForest&lt;DDataRecord&gt;(
    /// (t)=>
    /// {
    ///   //create and return a classifier such as a decision tree or perceptron
    ///   return new ID3();
    /// });
    /// classifier.Train(training_sample);
    /// 
    /// foreach(DDataRecord rec in testing_sample)
    /// {
    ///     string predicted_label = classifier.Predict(rec);
    /// }
    /// </summary>
    public class RandomForest<T> : Classifier<T, string>
        where T : DDataRecord 
    {
        protected double mPercentageDataUsage = 0.6667;
        protected ID3<T>[] mClassifiers = null;
        protected int mForestSize;
        protected int mFeatureSubSetSize;
        public delegate ID3<T> ClassifierGenerationMethod(int i);

        public int FeatureSubsetSize
        {
            get { return mFeatureSubSetSize; }
        }

        public RandomForest(ClassifierGenerationMethod generator, int forest_size = 800, int feature_subset_size=-1, double percentage_data_use = 0.6667)
        {
            mForestSize = forest_size;
            mPercentageDataUsage = percentage_data_use;
            mClassifiers = new ID3<T>[forest_size];
            for(int t=0; t < forest_size; ++t)
            {
                mClassifiers[t] = generator(t);
                mClassifiers[t].Forest = this;
            }
            mFeatureSubSetSize = -1;
        }

        public override void Train(IEnumerable<T> data_store)
        {
            int total_feature_count = data_store.First().FeatureCount;
            if (mFeatureSubSetSize == -1 || mFeatureSubSetSize < total_feature_count)
            {
                mFeatureSubSetSize = (int)System.Math.Sqrt(total_feature_count);
            }

            List<T> temp_samples = new List<T>();
            foreach (T rec in data_store)
            {
                temp_samples.Add(rec);
            }
            int sample_count = (int)(temp_samples.Count * mPercentageDataUsage);

            for (int t = 0; t < mForestSize; ++t)
            {
                List<T> new_training_sample = new List<T>();
                for (int i = 0; i < sample_count; ++i)
                {
                    int sample_index = RandomEngine.NextInt(sample_count);
                    new_training_sample.Add(temp_samples[sample_index]);
                }
                mClassifiers[t].Train(new_training_sample);
            }
        }

        public override string Predict(T rec)
        {
            Dictionary<string, int> votes = new Dictionary<string, int>();
            foreach (ID3<T> classifier in mClassifiers)
            {
                string predicted_class_variable_value = classifier.Predict(rec);
                if (votes.ContainsKey(predicted_class_variable_value))
                {
                    votes[predicted_class_variable_value]++;
                }
                else
                {
                    votes[predicted_class_variable_value] = 1;
                }
            }

            int highest_vote_count = 0;
            string highest_vote = null;
            foreach (string predicted_class_variable_value in votes.Keys)
            {
                int vote_count = votes[predicted_class_variable_value];
                if (highest_vote_count < vote_count)
                {
                    highest_vote_count = vote_count;
                    highest_vote = predicted_class_variable_value;
                }
            }

            return highest_vote;
        }
    }
}

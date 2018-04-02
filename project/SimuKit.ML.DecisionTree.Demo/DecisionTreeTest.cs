using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.DecisionTree.Demo
{
    using System.Xml;
    using System.IO;
    using ML.DecisionTree;
    using ML.Lang;

    public class DecisionTreeTest
    {
        public static List<DDataRecord> LoadSample()
        {
            XmlDocument doc = new XmlDocument();
            doc.Load("database.xml");

            List<DDataRecord> records = new List<DDataRecord>();

            XmlElement xml_root = doc.DocumentElement;
            foreach (XmlElement xml_level1 in xml_root.ChildNodes)
            {
                if (xml_level1.Name == "record")
                {
                    String outlook = xml_level1.Attributes["outlook"].Value;
                    string temperature = xml_level1.Attributes["temperature"].Value;
                    string humidity = xml_level1.Attributes["humidity"].Value;
                    String windy = xml_level1.Attributes["windy"].Value;
                    String class_label = xml_level1.Attributes["class"].Value;
                    DDataRecord rec = new DDataRecord();
                    rec["outlook"]=outlook;
                    rec["windy"]=windy;
                    rec["temperature"]=temperature;
                    rec["humidity"]=humidity;

                    rec.Label = class_label;
                    records.Add(rec);
                }
            }
            return records;
        }

        public static void RunC45()
        {
            List<DDataRecord> records = LoadSample();

            C45<DDataRecord> algorithm = new C45<DDataRecord>();
            algorithm.UpdateContinuousAttributes(records, "temperature");
            algorithm.UpdateContinuousAttributes(records, "humidity");
            algorithm.Train(records);
            //algorithm.RulePostPrune(records); //post pruning using cross valiation set

            Console.WriteLine("C4.5 Tree Built!");
		
		    for(int i=0; i<records.Count; i++)
		    {
			    DDataRecord rec=records[i];
                Console.WriteLine("rec: ");
                string[] feature_names = rec.FindFeatures();
                foreach(string feature_name in feature_names)
                {
                    Console.WriteLine(feature_name+" = " + rec[feature_name]);
                }
                Console.WriteLine("Label: " + rec.Label);
                Console.WriteLine("Predicted Label: " + algorithm.Predict(records[i]));
                Console.WriteLine();
		    }
        }

        public static void RunID3()
        {
            List<DDataRecord> X = LoadSample();

            //As ID3 does not support continuous value, must do manually conversion
            foreach (DDataRecord rec in X)
            {
                int temperature = int.Parse(rec["temperature"]);
                int humidity = int.Parse(rec["humidity"]);

                if (temperature < 75)
                {
                    rec["temperature"]="< 75";
                }
                else
                {
                    rec["temperature"]=">= 75";
                }
                if (humidity < 80)
                {
                    rec["humidity"]="< 80";
                }
                else
                {
                    rec["humidity"]=">= 80";
                }
            }

            ID3<DDataRecord> algorithm = new ID3<DDataRecord>();
            algorithm.Train(X);
            //algorithm.ErrorReducePrune(Xval); //error reduce prune using cross valiation set

            Console.WriteLine("ID3 Tree Built!");

            for (int i = 0; i < X.Count; i++)
            {
                DDataRecord rec = X[i];
                Console.WriteLine("rec: ");
                string[] feature_names = rec.FindFeatures();
                foreach(string feature_name in feature_names)
                {
                    Console.WriteLine(feature_name + " = " + rec[feature_name]);
                }
                Console.WriteLine("Label: " + rec.Label);
                Console.WriteLine("Predicted Label: " + algorithm.Predict(X[i]));
                Console.WriteLine();
            }

            algorithm.WriteToXml("ID3.xml");
            
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SimuKit.ML.DecisionTree.Helpers
{
    public static class CollectionExtensionMethods
    {
        public static HashSet<T> Clone<T>(this HashSet<T> obj)
        {
            HashSet<T> clone = new HashSet<T>();
            foreach (T rec in obj)
            {
                clone.Add(rec);
            }
            return clone;
        }

        public static void Shuffle<T>(this List<T> list)
        {
            if (list.Count == 0) return;

           int index=0;
           while (index < list.Count - 1)
           {
               int index2 = RandomEngine.NextInt(list.Count - index) + index;
               index2 = index2 % list.Count;
               T temp = list[index];
               list[index] = list[index2];
               list[index] = temp;
           }
        }
    }
}

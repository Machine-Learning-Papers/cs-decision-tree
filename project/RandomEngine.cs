using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DecisionTree
{
    public class RandomEngine
    {
        private static Random random = new Random();
        public static int NextInt(int upper)
        {
            return random.Next(upper);
        }
    }
}

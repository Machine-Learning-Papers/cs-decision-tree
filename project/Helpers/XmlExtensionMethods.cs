using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;

namespace SimuKit.ML.DecisionTree.Helpers
{
    public static class XmlExtensionMethods
    {
        public static XmlAttribute AppendAttribute(this XmlElement element, string name, object value)
        {
            if (value == null) return null;
            XmlAttribute attribute = element.OwnerDocument.CreateAttribute(name);
            attribute.Value = value.ToString();
            element.Attributes.Append(attribute);
            return attribute;
        }

        public static bool TryQueryStringAttribute(this XmlElement element, string name, out string value)
        {
            value = null;
            XmlAttribute attribute = element.Attributes[name];
            if (attribute != null)
            {
                value = attribute.Value;
                return true;
            }
            return false;
        }

        public static bool TryQueryDoubleAttribute(this XmlElement element, string name, out double value)
        {
            value = 0;
            XmlAttribute attribute = element.Attributes[name];
            if (attribute != null)
            {
                string attribute_value = attribute.Value;
                if (double.TryParse(attribute_value, out value))
                {
                    return true;
                }
            }
            return false;
        }

        public static bool TryQueryIntAttribute(this XmlElement element, string name, out int value)
        {
            value = 0;
            XmlAttribute attribute = element.Attributes[name];
            if (attribute != null)
            {
                string attribute_value = attribute.Value;
                if (int.TryParse(attribute_value, out value))
                {
                    return true;
                }
            }
            return false;
        }
    }
}

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    /// <summary>
    /// for testing the LSTM functions as they are coded
    /// </summary>
    class SampleGenerator
    {
        public SampleGenerator()
        {
            //Directory.CreateDirectory(".\\BaseInput");
            //Directory.CreateDirectory(".\\TargetOutput");

        }
        Random myRandom = new Random();
        double count1, count2, count3;
        double[] GenRandomINput(int vectLength)
        {

            double[] output = new double[vectLength];
            for (int i = 0; i < vectLength; i++)
            {
                double rand = myRandom.NextDouble();
                if (rand > 0.3) { output[i] = 1; }
                else { output[i] = 1; }
                output[i] = 1;
                //output[i] = myRandom.Next(0,2);
            }
            return output;
        }
        /// <summary>
        /// vector must have a length of 3.. checks if the count condition of each output has been reached, and gives a value
        /// of 1 for that output, otherwise ,the value is zero
        /// </summary>
        /// <param name="vector"></param>
        double[] CheckCount(double[] vector)
        {
            double out1 = -1, out2 = 0, out3 = 0;
            count1 += vector[0];
            // count2 += vector[1];
            // count3 += vector[2];

            //if (count1 > 5) {  out1 = 1;  count1=0;  }
            //if (count2 > 4) { out2 = 1; count2 = 0; }
            //if (count3 > 5) { out3 = 1; count3 = 0; }
            //out1 = -1;
            out1 = Math.Sin(count1 *(2 * 3.14)/20);
            // return new int[] { out1,out2,out3};
            return new double[] { out1 };//, out2, out3 };
        }
        public double[][] BaseT;
        public double[][] TOutput;
        public void GenerateData(int timeStep)
        {
            BaseT = new double[timeStep][];
            TOutput = new double[timeStep][];
            for (int t = 0; t < timeStep; t++)
            {
                double[] baseInput = GenRandomINput(1);
                double[] targetOutPut = CheckCount(baseInput);
                BaseT[t] = baseInput;
                TOutput[t] = targetOutPut;

                //Save(baseInput, ".\\BaseInput\\BaseInput" + t + ".txt");
                //Save(targetOutPut, ".\\TargetOutput\\TargetOutput" + t + ".txt");
            }
        }
        void Save(int[] data, string fileName)
        {
            StreamWriter sw = new StreamWriter(fileName);
            using (sw)
            {
                sw.WriteLine(data.Length);
                for (int i = 0; i < data.Length; i++)
                {
                    sw.WriteLine(data[i]);
                }
            }
        }

    }
}

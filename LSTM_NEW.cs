using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;

namespace Counter_Console
{
    class LSTM_NEW:LSTM_
    {
        public LSTM_NEW(NodeStruct myStruct, int[] hiddenNodes, int tStep):base(myStruct,hiddenNodes,tStep) { 
        }

        public double ComputeError2() {
            double error = 0;
            //iterate through all the timestep
            for (int i = 0; i < T; i++)
            {
                //get the final layer output for the current timestep
                Vector yVector = new Vector(HJ[L - 1, i]);
                //get the desired/ target output for the current timestep
                Vector tOutput = new Vector(TargetOutputT[i]);
                Vector errorArray = yVector - tOutput;
                errorArray = errorArray * errorArray;//square it

                int tt = errorArray.data.Length;
                for (int n = 0; n < errorArray.data.Length; n++)
                {
                    error += (errorArray.data[n]) / tt;
                }


            }
            error = error / T;
            return error;
        }
        

        public virtual void PropageForwardSingleTStep()
        {

            Stopwatch mywatch = new Stopwatch(); mywatch.Start();
            for (int t = 0; t < T; t++)
            {
                for (int l = 0; l < L; l++)
                {
                    ObtainNodes(l);//get the values of D and N
                                   //Console.WriteLine("d={0}:: n={1}",D,N);
                    InitialiseLayerWeights(l);
                    if (l == L - 1)
                    {
                        InitialiseAllVariables(l, t);
                        X = ArrayCopy(HJ[l - 1, t]);

                        An = elemAdd(Ba, MatrixMul(Wa, X));
                        A = Tanh(An);

                        H = ArrayCopy(An);//linear
                        HJ[l, t] = H;//save h
                    }

                    else
                    {
                        InitialiseAllVariables(l, t);

                        if (l == 0)
                        {
                            Console.WriteLine("Write the input variables for timestep: " + t);
                            double[] userInput = new double[N];
                            //N is the number of input nodes for a given layer
                            for (int n = 0; n < N; n++)
                            {
                                userInput[n] = double.Parse(Console.ReadLine());
                            }
                            X = userInput;
                        }
                        else {  X = GetInput(l, t);  }
                        

                            
                        ComputeVariables(l, t);//remove this from comment later


                        //compute layer output array
                        H = elemMul(O, Tanh(C));

                        SaveAllVariables(l, t);
                    }
                    


                }
                Vector HH = new Vector(H);
                DisplayOutput(HH, t);
            }

            
        }


        void DisplayOutput(Vector vector, int tstep)
        {
            Console.WriteLine("The output data for timestep {0} are: ", tstep);
            double[] vectorData = vector.data;
            foreach (double d in vectorData)
            {
                Console.Write(d + " , ");
            }
            Console.WriteLine("");
        }

    }
}

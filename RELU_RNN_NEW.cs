using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    class RELU_RNN_NEW : RELU_RNN
    {
        public RELU_RNN_NEW(int[] hiddenNodes, int TStep):base(hiddenNodes,TStep){  }
        /// <summary>
        /// propaate forward through all the layers, one timestap at a time
        /// Do not call ComputeError immdiately after using this ethod, as this method assues the NN has learnt, 
        /// and is to be used in actual practice
        /// </summary>
        public void PropageForwardSingleTStep()
        {
            //loop throug every timestep
            for(int t = 0; t < T; t++)
            {
                //for each timestep, loop through every layer
                for (int l = 0; l < L; l++)
                {
                    ObtainNodes(l);
                    InitLayerMarices(l);
                    InitVariableVectors(l, t);
                    if (l == 0)
                    {
                        Console.WriteLine("Write the input variables for timestep: " + t);
                        double[] userInput = new double[N];
                        //N is the number of input nodes for a given layer
                        for(int n = 0; n < N; n++)
                        {
                            userInput[n] = double.Parse(Console.ReadLine());
                        }
                        X.data = userInput;
                    }
                    else
                    {
                        X.data = GetInput(l, t);
                    }
                    
                    ComputeVariables2(l, t);
                    SaveAllVariables(l, t);

                    //extra 
                    
                }
                DisplayOutput(Y, t);
            }
        }
        //display the output data for each timestep to console
        void DisplayOutput(Vector vector, int tstep)
        {
            Console.WriteLine("The output data for timestep {0} are: ", tstep);
            double[] vectorData = vector.data;
            foreach(double d in vectorData)
            {
                Console.Write(d + " , ");
            }
            Console.WriteLine("");
        }
        /// <summary>
        /// error=summationof all (Y-Target output)^2
        /// </summary>
        public double ComputeError()
        {
            double error = 0;
            //iterate through all the timestep
            for(int i = 0; i < T; i++)
            {
                //get the final layer output for the current timestep
                Vector yVector = new Vector(YJ[L-1, i]);
                //get the desired/ target output for the current timestep
                Vector tOutput = new Vector(TargetOutputT[i]);
                Vector errorArray = yVector - tOutput;
                errorArray = errorArray*errorArray;//square it

                int tt = errorArray.data.Length;
                for(int n = 0; n < errorArray.data.Length; n++)
                {
                    error += (errorArray.data[n])/tt;
                }


            }
            error = error / T;
            return error;
        }
        /// <summary>
        /// performs the forward and backward propagation for a single timestep, and returns the 
        /// error averaged over all timesteps
        /// </summary>
        public double Train()
        {
            PropagateForward();
            PropagateBackward();
            double error = ComputeError();
            return error;
        }
        
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    public class RELU_RNN
    {
        public RELU_RNN(int[] Nodes, int tStep)
        {
            L = Nodes.Length - 1;
            T = tStep;
            incT = 2;
            lnRate = 0.005 / T;

            InitJgdArrays();
            nodes = Nodes;
            layerWeightsInitialised = new bool[L];
            M = new Methods();
        }
        public virtual void InitJgdArrays()
        {
            //the wweights
            WIJ = new double[L][,];
            UIJ = new double[L][,];
            //the bias
            BIJ = new double[L][];

            YJ = new double[L, T][];
            YNJ = new double[L, T][];
            XJ = new double[L, T][];

            DYJ = new double[L, T][];
            DYNJ = new double[L, T][];
            DXJ = new double[L, T][];

            BaseT = new double[T][];
            TargetOutputT = new double[T][];
        }


        public static Methods M = new Methods();
        /// <summary>
        /// the total number of layers
        /// </summary>
        public int L;//the total number of layers
        /// <summary>
        /// the number of output nodes for a given layer... must be set for each layer
        /// </summary>
        public int D;
        /// <summary>
        /// the number of input nodes for a given layer.... must be set for each layer
        /// </summary>
        public int N;
        /// <summary>
        /// the total number of timesteps...needs to figure out how to initialise this
        /// </summary>
        public int T;
        /// <summary>
        /// used for training in steadily increasing timesteps
        /// </summary>
        public int incT;
        /// <summary>
        /// the stochastic learning rate
        /// </summary>
        public double lnRate;
        /// <summary>
        /// each index represents whether the weights of a particular layer has been initialised
        /// </summary>
        public bool[] layerWeightsInitialised;

        /// <summary>
        /// the weight jagged arrays
        /// </summary>
        public double[][,] WIJ, UIJ;
        public double[][] BIJ;

        /// <summary>
        /// stores the number of nodes in each layer
        /// </summary>
        public int[] nodes;
        //thwe weights and biases
        public Matrix Wi, Ui, dWi, dUi;
        public Vector Bi, dBi;
        /// <summary>
        /// BaseT and TargetOutputT are the the base and target output for a single time series sample respectively
        /// </summary>
        public double[][] BaseT, TargetOutputT;
        /// <summary>
        /// the variable vectors
        /// </summary>
        public Vector Y, Yn, X;
        public Vector dY, dYn, dX;
        /// <summary>
        /// the corresponding variable jagged array
        /// </summary>
        public double[,][] YJ, YNJ, XJ;
        public double[,][] DYJ, DYNJ, DXJ;
        public Random myRandom = new Random();

        /// <summary>
        /// returns a dd X nn matrix filled with random values in the range 0-0.05
        /// </summary>
        /// <param name="dd"></param>
        /// <param name="nn"></param>
        /// <returns></returns>
        public double[,] GenRandomMatrix(int dd, int nn)
        {
            double[,] output = new double[dd, nn];

            for (int i = 0; i < dd; i++)
            {//rows
                for (int j = 0; j < nn; j++)
                {//columns
                    double sign = myRandom.NextDouble();
                    if (sign > 0.5) { output[i, j] = myRandom.NextDouble() * 0.05; }
                    else { output[i, j] = myRandom.NextDouble() * 0.05 * (-1); }
                    // output[i, j] = myRandom.NextDouble() * 0.05;//change to 0.05//previously 0.2
                    // output[i, j] = 1.0;//******************************remove thia
                }

            }
            return output;

        }
        /// <summary>
        /// returns a single column array of the required size containing values in the range 0-0.05.
        /// this is used for initialising bias weights
        /// </summary>
        /// <param name="length"></param>
        /// <returns></returns>
        public double[] GenRandomMatrix(int length)
        {
            double[] output = new double[length];
            for (int i = 0; i < length; i++)
            {
                output[i] = myRandom.NextDouble() * 0.05;//0.05
                //output[i] = 0.5;//remove this                                       
            }
            return output;
        }
        /// <summary>
        /// copies one matrix into another
        /// </summary>
        /// <returns></returns>
        public double[,] ArrayCopy(double[,] input)
        {
            double[,] output = new double[input.GetLength(0), input.GetLength(1)];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int j = 0; j < input.GetLength(1); j++)
                {
                    output[i, j] = input[i, j];
                }

            }
            return output;
        }
        /// <summary>
        /// copies one array into another
        /// </summary>
        /// <returns></returns>
        public double[] ArrayCopy(double[] input)
        {
            double[] output = new double[input.Length];
            for (int j = 0; j < input.Length; j++)
            {
                output[j] = input[j];
            }
            return output;
        }
        /// <summary>
        /// initialises the weights matrices of a specified layer.
        /// if the weights have been previously initialisd, it simply loads them from a file
        /// </summary>
        /// <param name="l"></param>
        public void InitLayerMarices(int l)
        {
            //if the weights of a paarticulaar layer have not been initialised, generate them
            if (!layerWeightsInitialised[l])
            {
                Wi.data = GenRandomMatrix(D, N);
                Ui.data = GenRandomMatrix(D, D);
                Bi.data = GenRandomMatrix(D);

                SaveWeights(l);
                layerWeightsInitialised[l] = true;
                Console.WriteLine("weights successfully generated");
            }
            //if they have been initialised, load them from the jagged arrays
            else
            {
                Wi.data = WIJ[l];
                Ui.data = UIJ[l];
                Bi.data = BIJ[l];
            }
        }
        /// <summary>
        /// store the weight data in the appropraite jagged arrays
        /// </summary>
        /// <param name="l"></param>
        public void SaveWeights(int l)
        {
            WIJ[l] = ArrayCopy(Wi.data);
            UIJ[l] = ArrayCopy(Ui.data);
            BIJ[l] = ArrayCopy(Bi.data);
        }

        /// <summary>
        /// performs the forward propagation through the counter network for all the timesteps
        /// </summary>
        public virtual void PropagateForward()
        {
            for (int l = 0; l < L; l++) 
            {
                ObtainNodes(l);
                InitLayerMarices(l);
                for (int t = 0; t < T; t++)
                {
                    InitVariableVectors(l, t);
                    X.data = GetInput(l, t);
                    ComputeVariables(l, t);
                    SaveAllVariables(l, t);
                }
            }
        }
        /// <summary>
        /// saves B, Bn, A, X in the corresponding jagged arrays
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void SaveAllVariables(int l, int t)
        {
            YJ[l, t] = Y.data;
            YNJ[l, t] = Yn.data;
            XJ[l, t] = X.data;

        }

        public void ComputeVariables(int l, int t)
        {
            Vector Yt_1;
            if (t > 0)
            {
                Yt_1.data = YJ[l, t - 1];
            }
            else
            {
                Yt_1 = new Vector(D);
            }
            Yn = Wi * X + Ui * Yt_1 + Bi;
            Y = M.ReLU(Yn);
        }

        /// <summary>
        /// gets the input data required to propagate forward through the particular layer and timestep
        ///.... l: the layer number
        ///.... t: the timestep
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        public double[] GetInput(int l, int t)
        {

            if (l == 0)
            {//if the input data required is for the first layer
                return BaseT[t];

            }
            else
            {//else load the output of the previous layer at that particular timestep
                return ArrayCopy(YJ[l - 1, t]);

            }

        }

        /// <summary>
        /// resets all variable vectors to xero length
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void InitVariableVectors(int l, int t)
        {
            Y.data = new double[0];
            Yn.data = new double[0];
            X.data = new double[0];

        }

        /// <summary>
        /// obtains the no of input nodes N, and the number of output nodes D for the specified layer
        /// from the file Nodes.txt
        /// </summary>
        /// <param name="l"></param>
        public void ObtainNodes(int l)
        {
            //string[] nodes = File.ReadAllLines("Nodes.txt");
            N = nodes[l];
            D = nodes[l + 1];
        }

        public virtual void PropagateBackward()
        {
            for (int l = L - 1; l >= 0; l--)
            {
                ObtainNodes(l);
                InitWeightGradients();
                InitLayerMarices(l);

                for (int t = T - 1; t >= 0; t--)
                {
                    InitVariableGradient();
                    LoadVariables(l, t);
                    dY = GetBackPropInput(l, t);
                    dYn = dY * M.ReLU_Prime(Yn);
                    dX = Wi.T() * dYn;

                    dWi += dYn ^ X;
                    dBi += dYn;
                    if (t > 0)
                    {
                        Vector yt_1 = new Vector(YJ[l, t - 1]);// y at t-1
                        dUi += dYn ^ yt_1;
                    }


                    SaveVariableGradient(l, t);
                }
                //compute new weights
                Wi += -1 * lnRate * dWi;
                Ui += -1 * lnRate * dUi;
                Bi += -1 * lnRate * dBi;
                SaveWeights(l);
            }
        }
        public void SaveVariableGradient(int l, int t)
        {
            DYJ[l, t] = dY.data;
            DYNJ[l, t] = dYn.data;
            DXJ[l, t] = dX.data;

        }

        public Vector GetBackPropInput(int l, int t)
        {
            Vector Del, dYnt1;
            //get del from the layer above or from the outptut
            if (l != L - 1) { Del.data = DXJ[l + 1, t]; }
            else { Del.data = M.elemSub(YJ[l, t], TargetOutputT[t]); }

            if (t < T - 1)
            {
                dYnt1.data = DYNJ[l, t + 1];
            }
            else
            {
                dYnt1.data = new double[D];
            }


            Vector output = Del + Ui.T() * dYnt1;

            return output;
        }

        /// <summary>
        /// loads the vectors At,Bt,Bnt,Xt
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void LoadVariables(int l, int t)
        {
            Y.data = YJ[l, t];
            Yn.data = YNJ[l, t];
            X.data = GetInput(l, t);
        }

        /// <summary>
        /// initialise all variable gradient vector data to zero length arrays
        /// </summary>
        public void InitVariableGradient()
        {
            dY.data = new double[0];
            dYn.data = new double[0];
            dX.data = new double[0];
        }

        /// <summary>
        /// intialises the matrix data of dWI, dUi, and dBi to zero vectors
        /// </summary>
        public void InitWeightGradients()
        {
            dWi.data = new double[D, N];
            dUi.data = new double[D, D];
            dBi.data = new double[D];
        }


    }
}

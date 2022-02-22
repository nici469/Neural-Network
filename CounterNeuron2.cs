using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Counter_Console
{
    public class CounterNeuron2
    {
        public static Methods M;
        public CounterNeuron2(int[] Nodes, int tStep)
        {
            L = Nodes.Length - 1;
            T = tStep;
            incT = 2;
            lnRate = 0.005 / T;

            InitJgdArrays();
            nodes = Nodes;
            layerWeightsInitialised = new bool[L];
            M = new Methods();
            isManyToOne = true;
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

        public virtual void InitJgdArrays()
        {
            //the wweights
            WIJ = new double[L][,];
            UIJ = new double[L][,];
            //the bias
            BIJ = new double[L][];

            AJ = new double[L, T][];
            XJ = new double[L, T][];
            BJ = new double[L, T][];
            BNJ = new double[L, T][];

            DAJ = new double[L, T][];
            DBJ = new double[L, T][];
            DBNJ = new double[L, T][];
            DYJ = new double[L, T][];

            BaseT = new double[T][];
            TargetOutputT = new double[T][];
        }
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
        /// <summary>
        /// bias
        /// </summary>
        public Vector Bi, dBi;
        /// <summary>
        /// BaseT and TargetOutputT are the the base and target output for a single time series sample respectively
        /// </summary>
        public double[][] BaseT, TargetOutputT;
        /// <summary>
        /// the variable vectors
        /// </summary>
        public Vector B, Bn, A, X;
        /// <summary>
        /// the variable gradients
        /// </summary>
        public Vector dB, dBn, dA, dX, dY;
        /// <summary>
        /// the vector of inputs from the previous layer
        /// </summary>
        public Vector Y;
        /// <summary>
        /// the corresponding variable jagged array
        /// </summary>
        public double[,][] BJ, BNJ, AJ, XJ;
        public double[,][] DBJ, DBNJ, DAJ, DXJ, DYJ;
        public Random myRandom = new Random(100);
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
                 // double sign = myRandom.Next();
                 //if (sign > 0.5) { output[i, j] = myRandom.NextDouble() * 0.5; }
                 //else { output[i, j] = myRandom.NextDouble() * 0.5*(-1); }
                    output[i, j] = myRandom.NextDouble() * 0.5;//change to 0.05//previously 0.2
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
                output[i] = myRandom.NextDouble() * 0.5;//0.05
                //output[i] = 0.5;//remove this                                        // output[i] = 0.5;//******************************remove thia
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
        public void PropagateForward()
        {
            for (int l = 0; l < L; l++)
            {
                ObtainNodes(l);
                InitLayerMarices(l);
                for (int t = 0; t < incT; t++)
                {
                    InitVariableVectors(l, t);
                    Y.data = GetInput(l, t);
                    ComputeVariables(l, t);
                    SaveAllVariables(l, t);
                }
            }
        }
        /// <summary>
        /// performs the forward propagation through the counter network for all the timesteps
        /// with an added tanh output layer
        /// </summary>
        public void PropagateForward2()
        {
            for (int l = 0; l < L; l++)
            {
                ObtainNodes(l);
                InitLayerMarices(l);
                if (l == L - 1)
                {//this is the tanh output layer
                    for(int t=0;t< incT; t++) {
                        InitVariableVectors(l,t);
                        Y.data = XJ[l - 1, t];
                        Bn = Wi * Y + Bi;
                        B = M.Tanh(Bn);
                        X = 1 * B;
                        SaveAllVariables(l, t);
                    }
                }
                else
                {
                    for (int t = 0; t < incT; t++)
                    {
                        InitVariableVectors(l, t);
                        Y.data = GetInput(l, t);
                        ComputeVariables(l, t);
                        SaveAllVariables(l, t);
                    }
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
            BJ[l, t] = B.data;
            BNJ[l, t] = Bn.data;
            AJ[l, t] = A.data;
            XJ[l, t] = X.data;

        }
        public void ComputeVariables(int l, int t)
        {
            Vector Xt_1;//it t=0, this is the initial state of X, else, it is the output at the previous timestep
            Vector H;//the layer output at t-1 timestep
            if (t == 0)
            {
                H = new Vector(D);
                Xt_1 = H - 1;//set to an initial state of -1
            }
            else
            {
                H.data = XJ[l, t - 1];
                Xt_1 = H;
            }

            Bn = Wi * Y + Ui * H + Bi;
            B = M.Sigmoid(Bn);

            A = Xt_1 * B;
            X = 1 + A;



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
                return ArrayCopy(XJ[l - 1, t]);

            }

        }

        /// <summary>
        /// resets all variable vectors to xero length
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void InitVariableVectors(int l, int t)
        {
            B.data = new double[0];
            Bn.data = new double[0];
            A.data = new double[0];
            Y.data = new double[0];
            X.data = new double[0];

        }

        public void PropagateBackward()
        {
            for (int l = L - 1; l >= 0; l--)
            {
                ObtainNodes(l);
                InitWeightGradients();
                InitLayerMarices(l);

                for (int t = incT - 1; t >= 0; t--)
                {
                    InitVariableGradient();
                    LoadVariables(l, t);
                    dX = GetBackPropInput(l, t);
                    ComputeVariableGrad(l, t);
                    SaveVariableGradient(l, t);
                }
                //compute new weights
                Wi += -1 * lnRate * dWi;
                Ui += -1 * lnRate * dUi;
                Bi += -1 * lnRate * dBi;
                SaveWeights(l);
            }
        }
        public bool isManyToOne;
        public void PropagateBackward2()
        {
            for (int l = L - 1; l >= 0; l--)
            {
                ObtainNodes(l);
                InitWeightGradients();
                InitLayerMarices(l);
                if (l == L - 1)
                {
                    for (int t = incT - 1; t >= 0; t--)
                    {
                        InitVariableGradient();
                        LoadVariables(l, t);
                        if (t < incT - 1 && isManyToOne)
                        {//if many-to-one and not the last time step, propagate zero gradient
                            dY.data = new double[N];//
                            SaveVariableGradient(l, t);
                            continue;
                        }
                        dB.data = M.elemSub(BJ[l, t], TargetOutputT[t]);
                        dBn = dB * (1 - (B * B));
                        dY = Wi.T() * dBn;

                        dWi += dBn ^ Y;
                        dBi += dBn;
                        SaveVariableGradient(l, t);                      
                    }
                }
                else
                {
                    for (int t = incT - 1; t >= 0; t--)
                    {
                        InitVariableGradient();
                        LoadVariables(l, t);
                        dX = GetBackPropInput(l, t);
                        ComputeVariableGrad(l, t);
                        SaveVariableGradient(l, t);
                    }
                }
                
                //compute new weights
                Wi += -1 * lnRate * dWi;
                Ui += -1 * lnRate * dUi;
                Bi += -1 * lnRate * dBi;
                SaveWeights(l);
            }
        }

        //void SaveWeights(int l) { }
        public void SaveVariableGradient(int l, int t)
        {
            DAJ[l, t] = dA.data;
            DBJ[l, t] = dB.data;
            DBNJ[l, t] = dBn.data;
            DYJ[l, t] = dY.data;

        }
        public Vector GetBackPropInput(int l, int t)
        {
            Vector Del, dAt1, dBt1, bt1, bnt1;
            //get del from the layer above or from the outptut
            if (l != L - 1) { Del.data = DYJ[l + 1, t]; }
            else { Del.data = M.elemSub(XJ[l, t], TargetOutputT[t]); }

            if (t < incT - 1)
            {
                dAt1.data = DAJ[l, t + 1];//da(t+1)
                bt1.data = BJ[l, t + 1];//b(t+1)
                bnt1.data = BNJ[l, t + 1];//bn(t+1)
            }
            else
            {
                dAt1 = new Vector(D);//zero vector
                bt1.data = new double[D];
                bnt1.data = new double[D];
            }


            Vector output = Del + dAt1 * bt1 + Ui.T() * bnt1;

            return output;
        }
        public void ComputeVariableGrad(int l, int t)
        {
            Vector xt_1, H;//x(t-1),H
            if (t > 0)
            {
                //load the out put of that layer at the previous timestep
                xt_1.data = XJ[l, t - 1];
                H = xt_1;
            }
            else
            {
                xt_1 = (new Vector(D)) - 1;//at t=1, the initial state of x(t-1) is -1
                H = new Vector(D);//zero vector
            }
            dA.data = ArrayCopy(dX.data);//da=dx
            dB = dA * xt_1;
            dBn = dB * M.SigmPrime(B);

            dY = Wi.T() * dBn;

            dWi += dBn ^ Y;
            dUi += dBn ^ H;
            dBi += dBn;

        }
        /// <summary>
        /// loads the vectors At,Bt,Bnt,Xt
        /// </summary>
        /// <param name="l"></param>
        /// <param name="t"></param>
        public void LoadVariables(int l, int t)
        {
            A.data = AJ[l, t];
            B.data = BJ[l, t];
            Bn.data = BNJ[l, t];
            X.data = XJ[l, t];
            Y.data = GetInput(l, t);
        }
        /// <summary>
        /// initialise all variable gradient vector data to zero length arrays
        /// </summary>
        public void InitVariableGradient()
        {
            dX.data = new double[0];
            dA.data = new double[0];
            dB.data = new double[0];
            dBn.data = new double[0];
            dY.data = new double[0];
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

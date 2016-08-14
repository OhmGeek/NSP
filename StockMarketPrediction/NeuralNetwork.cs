using System;

/// <summary>
/// Ryan Collins' Neural Network Class. Designed in 2014 for EPQ. Allows a feed forward network to be constructed, with RPROP Training.
/// </summary>
public class NeuralNetwork
{
    //NB: Deltas are the same as dErrorbydWeight.
   private int numOfInputNeurons;
    private int numOfHiddenNeurons;
    private int numOfOutputNeurons;

    //Weight Matrices (Arrays of the Weight Values, stored as Double Precision floating point numbers).
    private double[][,] networkWeights;    //Key: 0 is the input to hidden, 1 is the hidden to Output. Originally had the problem of using 1,2,...... but this causes errors.
    private double[][] biasWeights;
    
    //update values (delta ij) for RPROP
    private double[][,] networkUpdateValues;
    private double[][] biasUpdateValues;

    private double[][,] prevNetworkUpdateValues;
    private double[][] prevBiasUpdateValues;

    //Gradients
    private double[][] networkGradients;
    private double[][] prevNetworkGradients;

    private double[][,] dErrordWeightNetwork;
    private double[][] dErrordWeightBias;

    private double[][,] prevdErrordWeightNetwork;
    private double[][] prevdErrordWeightBias;

    //actual real weight changes
    private double[][,] weightChangesNetwork;
    private double[][] weightChangesBias;

    private double[][,] prevWeightChangesNetwork;
    private double[][] prevWeightChangesBias;

    private double[][] neuronSums = new double[2][];
    private double[][] neuronOutput = new double[2][];

    /// <summary>
    /// This creates the basic neural network structure.
    /// </summary>
    /// <param name="numInput">The number of input neurons.</param>
    /// <param name="numHidden">The number of hidden neurons</param>
    /// <param name="numOutput">The number of output neurons</param>
    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
       // Console.WriteLine("In the new routine.");


        //specify number of neurons in each layer.
        this.numOfInputNeurons = numInput;
        this.numOfHiddenNeurons = numHidden;
        this.numOfOutputNeurons = numOutput;



        //Large Weight Matrix Creation.
        this.networkWeights = new double[2][,];
        this.networkWeights[0] = new double[numInput, numHidden]; //create input to hidden weight matrix.
        this.networkWeights[1] = new double[numHidden, numOutput]; //create hidden to output weight matrix.

        //Bias Matrix Creation
        this.biasWeights = new double[2][];
        this.biasWeights[0] = new double[numHidden]; //create space to store the bias weights to hidden layer.
        this.biasWeights[1] = new double[numOutput]; //create space to store teh bias weights to the output layer.



        SetupRPROPMatrices(); //this creates the space in memory for all the RPROP matrices.
        RandomiseWeights();








    }

    /// <summary>
    /// This creates a neural network from a previously saved state.
    /// </summary>
    /// <param name="filename">The whole file path to read the setup file from.</param>
    public NeuralNetwork(string filename)
    {

        System.IO.BinaryReader reader = null;
        System.IO.FileStream fileStream = null;

        try
        {
            fileStream = new System.IO.FileStream(filename, System.IO.FileMode.Open);
            reader = new System.IO.BinaryReader(fileStream);

            this.numOfInputNeurons = reader.ReadInt32();
            this.numOfHiddenNeurons = reader.ReadInt32();
            this.numOfOutputNeurons = reader.ReadInt32();


            //Large Weight Matrix Creation.
            this.networkWeights = new double[2][,];
            this.networkWeights[0] = new double[numOfInputNeurons, numOfHiddenNeurons]; //create input to hidden weight matrix.
            this.networkWeights[1] = new double[numOfHiddenNeurons, numOfOutputNeurons]; //create hidden to output weight matrix.

            //Bias Matrix Creation
            this.biasWeights = new double[2][];
            this.biasWeights[0] = new double[numOfHiddenNeurons]; //create space to store the bias weights to hidden layer.
            this.biasWeights[1] = new double[numOfOutputNeurons]; //create space to store teh bias weights to the output layer.

            for (int i = 0; i < numOfInputNeurons; i++)
                for (int j = 0; j < numOfHiddenNeurons; j++)
                    this.networkWeights[0][i, j] = reader.ReadDouble();

            for (int j = 0; j < numOfHiddenNeurons; j++)
                for (int k = 0; k < numOfOutputNeurons; k++)
                    this.networkWeights[1][j, k] = reader.ReadDouble();

            for (int j = 0; j < numOfHiddenNeurons; j++)
                this.biasWeights[0][j] = reader.ReadDouble();

            for (int k = 0; k < numOfOutputNeurons; k++)
                this.biasWeights[1][k] = reader.ReadDouble();

        }
        catch
        {
            

        }
        finally
        {
            if (fileStream != null)
                fileStream.Close();
            if (reader != null)
                reader.Close();

        }
        

    }

    /// <summary>
    /// This gets the outputs from the neural network.
    /// </summary>
    /// <returns>A 1D double array containing the output values from the output neurons.</returns>
    public double[] GetOutputs()
    {
        return neuronOutput[1];
    }



  
    /// <summary>
    /// This trains the neural network using the Resilient Propagation Algorithm
    /// </summary>
    /// <param name="TrainingInputData">A Jagged 2D Array containing the input data.</param>
    /// <param name="TrainingOutputData">A jagged 2D Array containing the outputs corresponding to the inputs.</param>
    /// <param name="numberOfRecords">The number of records in the training data.</param>
    /// <param name="eta_minus">The minimum learning constant (for decreasing descent velocity)</param>
    /// <param name="eta_plus">The maximum learning constant (for increasing the descent velocity)</param>
    /// <param name="maxNumberOfEpochs">The maximum number of epochs that one would like to run.</param>
    /// <param name="errorThreshold">The error at which training will stop.</param>
    /// <returns>The final global error. </returns>
    public double TrainByRPROP(double[][] TrainingInputData, double[][] TrainingOutputData, int numberOfRecords, double eta_minus, double eta_plus, int maxNumberOfEpochs,double errorThreshold)
    {
        int numberOfEpochs = 0;
        double globalError = 99999;
  
        do {
            globalError = 0.0;
            NewEpoch(); //after every epoch, zero gradients so that we can perform batch rprop.
            //Console.WriteLine("Epoch number: " + numberOfEpochs);

            globalError += CalculateGradientsAndDeltas(TrainingInputData, TrainingOutputData, numberOfRecords);

            //NOW THAT WE HAVE DONE THIS CORRECTLY, START TO MODIFY THE WEIGHTS:

            //1. Calculate teh update values for each of the individual weights:
            for (int i = 0; i < numOfInputNeurons; i++)
            {
                for (int j = 0; j < numOfHiddenNeurons; j++)
                {

                    if (dErrordWeightNetwork[0][i, j] * prevdErrordWeightNetwork[0][i, j] > 0)
                        networkUpdateValues[0][i, j] = eta_plus * prevNetworkUpdateValues[0][i, j];
                    else if (dErrordWeightNetwork[0][i, j] * prevdErrordWeightNetwork[0][i, j] < 0)
                        networkUpdateValues[0][i, j] = eta_minus * prevNetworkUpdateValues[0][i, j];
                    else
                        networkUpdateValues[0][i, j] = prevNetworkUpdateValues[0][i, j];



                    //weight updates:
                    if (dErrordWeightNetwork[0][i, j] * prevdErrordWeightNetwork[0][i, j] < 0)
                    {
                        weightChangesNetwork[0][i, j] = -prevWeightChangesNetwork[0][i, j];
                        dErrordWeightNetwork[0][i, j] = 0.0;
                    }
                    else if (dErrordWeightNetwork[0][i, j] > 0)
                        weightChangesNetwork[0][i, j] = -networkUpdateValues[0][i, j];
                    else if (dErrordWeightNetwork[0][i, j] < 0)
                        weightChangesNetwork[0][i, j] = networkUpdateValues[0][i, j];
                    else
                        weightChangesNetwork[0][i, j] = 0.0;



                    //now actually do the weight modification:
                    networkWeights[0][i, j] += weightChangesNetwork[0][i, j];

                }

            }

            //bias for hidden:

            for (int j = 0; j < numOfHiddenNeurons; j++)
            {


                if (dErrordWeightBias[0][j] * prevdErrordWeightBias[0][j] > 0)
                    biasUpdateValues[0][j] = eta_plus * prevBiasUpdateValues[0][j];
                else if (dErrordWeightBias[0][j] * prevdErrordWeightBias[0][j] < 0)
                    biasUpdateValues[0][j] = eta_minus * prevBiasUpdateValues[0][j];
                else
                    biasUpdateValues[0][j] = prevBiasUpdateValues[0][j];

                //weight updates:
                if (dErrordWeightBias[0][j] * prevdErrordWeightBias[0][j] < 0)
                {
                    weightChangesBias[0][j] = -prevWeightChangesBias[0][j];
                    dErrordWeightBias[0][j] = 0.0;
                }
                else if (dErrordWeightBias[0][j] > 0)
                    weightChangesBias[0][j] = -biasUpdateValues[0][j];
                else if (dErrordWeightBias[0][j] < 0)
                    weightChangesBias[0][j] = biasUpdateValues[0][j];
                else
                    weightChangesBias[0][j] = 0.0;

                biasWeights[0][j] += weightChangesBias[0][j];

            }

            for (int j = 0; j < numOfHiddenNeurons; j++)
            {
                for (int k = 0; k < numOfOutputNeurons; k++)
                {



                    if (dErrordWeightNetwork[1][j, k] * prevdErrordWeightNetwork[1][j, k] > 0)
                        networkUpdateValues[1][j, k] = eta_plus * prevNetworkUpdateValues[1][j, k];
                    else if (dErrordWeightNetwork[1][j, k] * prevdErrordWeightNetwork[1][j, k] < 0)
                        networkUpdateValues[1][j, k] = eta_minus * prevNetworkUpdateValues[1][j, k];
                    else
                        networkUpdateValues[1][j, k] = prevNetworkUpdateValues[1][j, k];

                    //weight updates:
                    if (dErrordWeightNetwork[1][j, k] * prevdErrordWeightNetwork[1][j, k] < 0)
                    {
                        weightChangesNetwork[1][j, k] = -prevWeightChangesNetwork[1][j, k];
                        dErrordWeightNetwork[1][j, k] = 0.0;
                    }
                    else if (dErrordWeightNetwork[1][j, k] > 0)
                        weightChangesNetwork[1][j, k] = -networkUpdateValues[1][j, k];
                    else if (dErrordWeightNetwork[1][j, k] < 0)
                        weightChangesNetwork[1][j, k] = networkUpdateValues[1][j, k];
                    else
                        weightChangesNetwork[1][j, k] = 0.0;



                    //now actually do the weight modification:
                    networkWeights[1][j, k] += weightChangesNetwork[1][j, k];


                }

            }

            for (int k = 0; k < numOfOutputNeurons; k++)
            {
                if (dErrordWeightBias[1][k] * prevdErrordWeightBias[1][k] > 0)
                    biasUpdateValues[1][k] = eta_plus * prevBiasUpdateValues[1][k];
                else if (dErrordWeightBias[1][k] * prevdErrordWeightBias[1][k] < 0)
                    biasUpdateValues[1][k] = eta_minus * prevBiasUpdateValues[1][k];
                else
                    biasUpdateValues[1][k] = prevBiasUpdateValues[1][k];

                //weight updates:
                if (dErrordWeightBias[1][k] * prevdErrordWeightBias[1][k] < 0)
                {
                    weightChangesBias[1][k] = -prevWeightChangesBias[1][k];
                    dErrordWeightBias[1][k] = 0.0;
                }
                else if (dErrordWeightBias[1][k] > 0)
                    weightChangesBias[1][k] = -biasUpdateValues[1][k];
                else if (dErrordWeightBias[1][k] < 0)
                    weightChangesBias[1][k] = biasUpdateValues[1][k];
                else
                    weightChangesBias[1][k] = 0.0;

                biasWeights[1][k] += weightChangesBias[1][k];
            }



            globalError /= numberOfRecords;

            //Console.WriteLine("The error for this epoch is: " + globalError);
            numberOfEpochs += 1;

        } while ((numberOfEpochs<maxNumberOfEpochs) && (globalError > errorThreshold));

        return globalError;
    }

    /// <summary>
    /// This randomises the weights of the neural network.
    /// </summary>
    public void RandomiseWeights()
    {
        const double updateAtZero = 0.05;
        if (this.networkWeights == null)
            throw new Exception("Main network matrix is null.");
        if (this.biasWeights == null)
            throw new Exception("Bias Network Matrix is null.");

        System.Random randomGenerator = new System.Random(DateTime.Now.Millisecond); //time seeded random number generator, to make the numbers more randomly selected.

        //input to hidden:
        for (int i = 0; i < numOfInputNeurons; i++)
            for (int j = 0; j < numOfHiddenNeurons; j++)
                this.networkWeights[0][i, j] = randomGenerator.NextDouble() - 0.5d; //random number between -0.5 and 0.5 inclusive. subtracting 0.5 because the generator generates between 0.0 and 1.0

        //hidden to output
        for (int i = 0; i < numOfHiddenNeurons; i++)
            for (int j = 0; j < numOfOutputNeurons; j++)
                this.networkWeights[1][i, j] = randomGenerator.NextDouble() - 0.5d; //random number between -0.5 and 0.5 inclusive. subtracting 0.5 because the generator generates between 0.0 and 1.0    

        //bias weights hidden:

        for (int j = 0; j < numOfHiddenNeurons; j++)
            this.biasWeights[0][j] = randomGenerator.NextDouble() - 0.5d;

        //Bias Weights Output.
        for (int k = 0; k < numOfOutputNeurons; k++)
            this.biasWeights[1][k] = randomGenerator.NextDouble() - 0.5d;


        for (int i = 0; i < numOfInputNeurons; i++)
            for (int j = 0; j < numOfHiddenNeurons; j++)
            {
                this.networkUpdateValues[0][i, j] = updateAtZero;
                this.dErrordWeightNetwork[0][i, j] = randomGenerator.NextDouble() - 0.5;

            }

        for (int j = 0; j < numOfHiddenNeurons; j++)
            for (int k = 0; k < numOfOutputNeurons; k++)
            {

                this.networkUpdateValues[1][j, k] = updateAtZero;
                this.dErrordWeightNetwork[1][j, k] = randomGenerator.NextDouble() - 0.5;

            }

        for (int j = 0; j < numOfHiddenNeurons; j++)
        {
            this.biasUpdateValues[0][j] = updateAtZero;
            this.dErrordWeightBias[0][j] = randomGenerator.NextDouble() - 0.5;
        }

        for (int k = 0; k < numOfOutputNeurons; k++)
        {
            this.biasUpdateValues[1][k] = updateAtZero;
            this.dErrordWeightBias[1][k] = randomGenerator.NextDouble() - 0.5;
        }

    }

 
    /// <summary>
    /// This calculates the outputs and the corresponding output sums. Call this first, and then afterwards get the outputs for yielding the NN output values.
    /// </summary>
    /// <param name="inputs">A 1D Array containing values to be input into the input layer.</param>
    public void CalculateOutputs(double[] inputs)
    {
        //validation check
        if (inputs.Length != numOfInputNeurons)
            throw new Exception("Unequal number of inputs and neurons for calculation. check number of inputs are correct, or number of input neurons is sufficient.");




        neuronSums[0] = new double[numOfHiddenNeurons];
        neuronSums[1] = new double[numOfOutputNeurons];

        neuronOutput[0] = new double[numOfHiddenNeurons];
        neuronOutput[1] = new double[numOfOutputNeurons];


        //SET VALUES TO ZERO. DON't ASSUME THIS AUTOMATICALLY HAPPENS.
        for (int hiddenCounter = 0; hiddenCounter < numOfHiddenNeurons; hiddenCounter++)
        {
            neuronSums[0][hiddenCounter] = 0.0;
            neuronOutput[0][hiddenCounter] = 0.0;
        }

        for (int outputCounter = 0; outputCounter < numOfOutputNeurons; outputCounter++)
        {
            neuronSums[1][outputCounter] = 0.0;
            neuronOutput[1][outputCounter] = 0.0;
        }


        //Input to Hidden. This also features bias addition.
        for (int j = 0; j < numOfHiddenNeurons; j++)
        {
            for (int i = 0; i < numOfInputNeurons; i++)
            {
                neuronSums[0][j] += inputs[i] * networkWeights[0][i, j];
            }
            neuronSums[0][j] += biasWeights[0][j];
            neuronOutput[0][j] = this.ThresholdFunction(neuronSums[0][j]);
        }


        //Hidden to Output 
        for (int k = 0; k < numOfOutputNeurons; k++)
        {
            for (int j = 0; j < numOfHiddenNeurons; j++)
            {
                neuronSums[1][k] += neuronOutput[0][j] * networkWeights[1][j, k];
            }
            neuronSums[1][k] += biasWeights[1][k];
            neuronOutput[1][k] = this.ThresholdFunction(neuronSums[1][k]);
        }
    }

    public int InputNeuronCount()
    {
        return this.numOfInputNeurons;
    }
    public int OutputNeuronCount()
    {
        return this.numOfOutputNeurons;
    }
    /// <summary>
    /// This saves the neural network to a file for later access.
    /// </summary>
    /// <param name="filePath">Filename and path to be saved to.</param>
    public void SaveNetToFile(string filePath)
    {
        System.IO.BinaryWriter writer = null;
        System.IO.FileStream fileStream = null;
        try
        {
            fileStream = new System.IO.FileStream(filePath, System.IO.FileMode.Create);
            writer = new System.IO.BinaryWriter(fileStream);

            //firstly, write the network structure (Input, Hidden and Output)
            writer.Write(this.numOfInputNeurons);
            writer.Write(this.numOfHiddenNeurons);
            writer.Write(this.numOfOutputNeurons);

            //now write the input to hidden weights:
            for (int i = 0; i < numOfInputNeurons; i++)
                for (int j = 0; j < numOfHiddenNeurons; j++)
                    writer.Write(this.networkWeights[0][i, j]);

            //now write the hidden to output weights:
            for (int j = 0; j < numOfHiddenNeurons; j++)
                for (int k = 0; k < numOfOutputNeurons; k++)
                    writer.Write(this.networkWeights[1][j, k]);

            //now write the hidden bias weights
            for (int j = 0; j < numOfHiddenNeurons; j++)
                writer.Write(this.biasWeights[0][j]);

            for (int k = 0; k < numOfOutputNeurons; k++)
                writer.Write(this.biasWeights[1][k]);
        }
        catch (Exception ex)
        {
            throw ex;
        }
        finally
        {
            if (fileStream != null)
                fileStream.Close();
            if (writer != null)
                writer.Close();

        }






    }


    //Private Functions - These aren't directly accessed by the programmer. Some also exist to make the code easier to understand.
    private void NewEpoch()
    {

        //prevGrads = current
        //prevdErrordWeights = current
        // prev weightchangesnetwork = current
        //prev update values = old update values




        for (int i = 0; i < numOfInputNeurons; i++)
            for (int j = 0; j < numOfHiddenNeurons; j++)
            {
                prevdErrordWeightNetwork[0][i, j] = dErrordWeightNetwork[0][i, j];
                dErrordWeightNetwork[0][i, j] = 0.0;


                prevWeightChangesNetwork[0][i, j] = weightChangesNetwork[0][i, j];
                weightChangesNetwork[0][i, j] = 0.0;



                prevNetworkUpdateValues[0][i, j] = networkUpdateValues[0][i, j];
                networkUpdateValues[0][i, j] = 0.0;
            }

        for (int j = 0; j < numOfHiddenNeurons; j++)
        {
            prevNetworkGradients[0][j] = networkGradients[0][j];
            networkGradients[0][j] = 0.0;


            prevWeightChangesBias[0][j] = weightChangesBias[0][j];
            weightChangesBias[0][j] = 0.0;


            prevBiasUpdateValues[0][j] = biasUpdateValues[0][j];
            biasUpdateValues[0][j] = 0.0;


            prevdErrordWeightBias[0][j] = dErrordWeightBias[0][j];
            dErrordWeightBias[0][j] = 0.0;

        }

        for (int j = 0; j < numOfHiddenNeurons; j++)
            for (int k = 0; k < numOfOutputNeurons; k++)
            {
                prevdErrordWeightNetwork[1][j, k] = dErrordWeightNetwork[1][j, k];
                dErrordWeightNetwork[1][j, k] = 0.0;

                prevWeightChangesNetwork[1][j, k] = weightChangesNetwork[1][j, k];
                weightChangesNetwork[1][j, k] = 0.0;



                prevNetworkUpdateValues[1][j, k] = networkUpdateValues[1][j, k];
                networkUpdateValues[1][j, k] = 0.0;



            }

        for (int k = 0; k < numOfOutputNeurons; k++)
        {
            prevNetworkGradients[1][k] = networkGradients[1][k];
            networkGradients[1][k] = 0.0;

            prevWeightChangesBias[1][k] = weightChangesBias[1][k];
            weightChangesBias[1][k] = 0.0;

            prevBiasUpdateValues[1][k] = biasUpdateValues[1][k];
            biasUpdateValues[1][k] = 0.0;

            prevdErrordWeightBias[1][k] = dErrordWeightBias[1][k];
            dErrordWeightBias[1][k] = 0.0;
        }

    }

    private double ThresholdFunction(double x)
    {
        return Math.Tanh(x);
    }

    private double ThresholdFunctionDerivative(double y)
    {
        return (1 - Math.Pow(y, 2));
    }

    private double CalculateGradientsAndDeltas(double[][] TrainingInputData, double[][] TrainingOutputData, int numberOfRecords)
    {
        double globalError = 0.0;

        
        for (int dataCounter = 0; dataCounter < numberOfRecords; dataCounter++)
        {
            double tempError = 0;
            CalculateOutputs(TrainingInputData[dataCounter]);
            //calculate the output gradients:

            for (int k = 0; k < this.numOfOutputNeurons; k++)
            {


                this.networkGradients[1][k] = (TrainingOutputData[dataCounter][k] - this.neuronOutput[1][k]) * ThresholdFunctionDerivative(this.neuronOutput[1][k]);
                tempError += (TrainingInputData[dataCounter][k] - this.neuronOutput[1][k]) * (TrainingInputData[dataCounter][k] - this.neuronOutput[1][k]);
            }

            //calculate the hidden gradients.
            for (int j = 0; j < this.numOfHiddenNeurons; j++)
            {
                double sum = 0.0;

                for (int k = 0; k < this.numOfOutputNeurons; k++)
                {
                    sum += networkGradients[1][k] * networkWeights[1][j, k];
                }
                networkGradients[0][j] = ThresholdFunctionDerivative(neuronOutput[0][j]) * sum;
            }

            //calculate the input-hidden deltas (sum these over the whole training set.)

            for (int i = 0; i < numOfInputNeurons; i++)
                for (int j = 0; j < numOfHiddenNeurons; j++)
                {
                    dErrordWeightNetwork[0][i, j] += -TrainingInputData[dataCounter][i] * networkGradients[0][j];
                    //Console.WriteLine("Delta for Hidden " + i + " to " + j + " is " + dErrordWeightNetwork[0][i, j]);
                }
            //calculate the hidden-output deltas (sum these over all the training set).

            for (int j = 0; j < numOfHiddenNeurons; j++)
                for (int k = 0; k < numOfOutputNeurons; k++)
                    dErrordWeightNetwork[1][j, k] += -neuronOutput[0][j] * networkGradients[1][k];

            //input-hidden bias neurons deltas:

            for (int j = 0; j < numOfHiddenNeurons; j++)
                dErrordWeightBias[0][j] += -1 * networkGradients[0][j];

            //hidden-output bias neurons deltas:
            for (int k = 0; k < numOfOutputNeurons; k++)
                dErrordWeightBias[1][k] += -1 * networkGradients[1][k];

            globalError += tempError;
        }











        return globalError;//returns the global error.
    }

    private void SetupRPROPMatrices()
    {




        //do this for the update values:
        this.networkUpdateValues = new double[2][,];
        this.networkUpdateValues[0] = new double[this.numOfInputNeurons, this.numOfHiddenNeurons];
        this.networkUpdateValues[1] = new double[this.numOfHiddenNeurons, this.numOfOutputNeurons];

        this.prevNetworkUpdateValues = new double[2][,];
        this.prevNetworkUpdateValues[0] = new double[this.numOfInputNeurons, this.numOfHiddenNeurons];
        this.prevNetworkUpdateValues[1] = new double[this.numOfHiddenNeurons, this.numOfOutputNeurons];


        //now repeat again for the bias update values:
        this.biasUpdateValues = new double[2][];
        this.biasUpdateValues[0] = new double[this.numOfHiddenNeurons];
        this.biasUpdateValues[1] = new double[this.numOfOutputNeurons];

        this.prevBiasUpdateValues = new double[2][];
        this.prevBiasUpdateValues[0] = new double[this.numOfHiddenNeurons];
        this.prevBiasUpdateValues[1] = new double[this.numOfOutputNeurons];
        //---------------------------------------------
        this.dErrordWeightNetwork = new double[2][,];
        this.dErrordWeightNetwork[0] = new double[numOfInputNeurons, numOfHiddenNeurons];
        this.dErrordWeightNetwork[1] = new double[numOfHiddenNeurons, numOfOutputNeurons];

        this.prevdErrordWeightNetwork = new double[2][,];
        this.prevdErrordWeightNetwork[0] = new double[numOfInputNeurons, numOfHiddenNeurons];
        this.prevdErrordWeightNetwork[1] = new double[numOfHiddenNeurons, numOfOutputNeurons];

        this.dErrordWeightBias = new double[2][];
        this.dErrordWeightBias[0] = new double[numOfHiddenNeurons];
        this.dErrordWeightBias[1] = new double[numOfOutputNeurons];

        this.prevdErrordWeightBias = new double[2][];
        this.prevdErrordWeightBias[0] = new double[numOfHiddenNeurons];
        this.prevdErrordWeightBias[1] = new double[numOfOutputNeurons];
        //---------------------------------------------
        this.networkGradients = new double[2][];
        this.networkGradients[0] = new double[numOfHiddenNeurons];
        this.networkGradients[1] = new double[numOfOutputNeurons];

        this.prevNetworkGradients = new double[2][];
        this.prevNetworkGradients[0] = new double[numOfHiddenNeurons];
        this.prevNetworkGradients[1] = new double[numOfOutputNeurons];
        //----------------------------------------------

        weightChangesNetwork = new double[2][,];
        weightChangesNetwork[0] = new double[numOfInputNeurons, numOfHiddenNeurons];
        weightChangesNetwork[1] = new double[numOfHiddenNeurons, numOfOutputNeurons];

        weightChangesBias = new double[2][];
        weightChangesBias[0] = new double[numOfHiddenNeurons];
        weightChangesBias[1] = new double[numOfOutputNeurons];
        //----------------------------------------------
        prevWeightChangesNetwork = new double[2][,];
        prevWeightChangesNetwork[0] = new double[numOfInputNeurons, numOfHiddenNeurons];
        prevWeightChangesNetwork[1] = new double[numOfHiddenNeurons, numOfOutputNeurons];

        prevWeightChangesBias = new double[2][];
        prevWeightChangesBias[0] = new double[numOfHiddenNeurons];
        prevWeightChangesBias[1] = new double[numOfOutputNeurons];

    }

}

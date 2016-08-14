using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using Microsoft.VisualBasic;
namespace StockMarketPrediction
{
    class Program
    {
        public static int input;
        public static int hidden;
        public static int output;
        public static NeuralNetwork stockPredictor;

        public static void Normalize(List<double[]> targetInputs, List<double[]> targetOutputs, List<StockData> stockMarketData) {
             for (int counter = 1; counter < stockMarketData.Count; counter++)
                stockMarketData[counter].Change = (stockMarketData[counter].Value - stockMarketData[counter - 1].Value) / (stockMarketData[counter - 1].Value);
            int j = 0;
            for (j = 1; j < stockMarketData.Count - 10; j++)
            {
                double[] inputs = new double[10];
                double[] outputs = new double[1];
                for (int i = 0; i < 10; i++)
                {
                    inputs[i] = stockMarketData[j + i].Change * 100;
                    // Console.WriteLine(inputs[i]);
                }
                outputs[0] = stockMarketData[j + 10].Change * 100;
                targetInputs.Add(inputs);
                targetOutputs.Add(outputs);
            }


        }
        public static void Train()
        {
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("Number of Input Neurons: ");
            input = Int32.Parse(Console.ReadLine());

            Console.Write("Number of Hidden Neurons: ");
            hidden = Int32.Parse(Console.ReadLine());

            Console.Write("Number of Output Neurons: ");
            output = Int32.Parse(Console.ReadLine());
            Console.WriteLine();
            Console.WriteLine("Creating a " + input + "-" + hidden + "-" + output + " Network");
            stockPredictor = new NeuralNetwork(input, hidden, output);


            Console.WriteLine("Please enter filename where data is found");
            string filename = Console.ReadLine();

            List<StockData> stockMarketData = LoadStockMarketData(filename);

            Console.WriteLine("The number of items loaded is " + stockMarketData.Count);


            //clone stock market data (make copy)

            //Console.WriteLine("Please enter filename where interest rate is found:");
            // string filenameForInterestRate = Console.ReadLine();

            //Dictionary<DateTime, Double> interestRateData = new Dictionary<DateTime, Double>();
            //LoadInterestRateData(filenameForInterestRate, ref interestRateData);

            //PERCENTAGE ERROR IS X-Y/Y

            List<double[]> targetInputs = new List<double[]>();
            List<double[]> targetOutputs = new List<double[]>();


            List<double[]> testTargetInputs = new List<double[]>();
            List<double[]> testTargetOutputs = new List<double[]>();

            Normalize(targetInputs, targetOutputs, stockMarketData);

            Console.WriteLine("Enter the fraction of test data wanted. e.g. 0.3 is 30% test data, and 70% training data.");
            double testDataFraction = Double.Parse(Console.ReadLine());
            Console.WriteLine();

            int numberOfTestItems = Convert.ToInt32(targetInputs.Count * testDataFraction);
            Random randGen = new Random();

            if (targetInputs.Count == 0)
                throw new Exception("Number of records is zero.");


            for (int i = 1; i < numberOfTestItems; i++)
            {
                int randomIndex = randGen.Next(0, targetInputs.Count);
                double[] tempInputs = targetInputs[randomIndex];
                double[] tempOutputs = targetOutputs[randomIndex];
                targetInputs.RemoveAt(randomIndex);
                targetOutputs.RemoveAt(randomIndex);
                testTargetInputs.Add(tempInputs);
                testTargetOutputs.Add(tempOutputs);

            }




            double[][] inputToNetwork = targetInputs.ToArray();
            double[][] outputFromNetwork = targetOutputs.ToArray();


            Console.WriteLine("The training set contains " + targetInputs.Count + " records. The test set contains " + testTargetInputs.Count + " records");
            Console.WriteLine();
            Console.WriteLine("First inputs are: " + inputToNetwork[0][0] + " " + inputToNetwork[0][1] + "...");
            Console.WriteLine();
            Console.WriteLine("Output from this is: " + outputFromNetwork[0][0]);
            Console.WriteLine();
            Console.WriteLine("Please enter constants for RPROP:");
            Console.Write("Eta Minus: ");
            double etaminus = Double.Parse(Console.ReadLine());
            Console.Write("Eta Plus: ");
            double etaplus = Double.Parse(Console.ReadLine());
            Console.Write("Epoch Max: ");
            int maxEpochs = Int32.Parse(Console.ReadLine());
            Console.WriteLine();

            Console.WriteLine("Press any key to start training...");
            Console.ReadLine();
            stockPredictor.TrainByRPROP(inputToNetwork, outputFromNetwork, targetInputs.Count, etaminus, etaplus, maxEpochs, 0.1);
            Console.WriteLine("The network has now been trained");
            Console.WriteLine("Press any key to continue...");
            Console.ReadLine();
            Console.WriteLine();
            Console.WriteLine("Testing will now commence:");
            Console.WriteLine();

            double totalError = 0.0;
            for (int i = 1; i < numberOfTestItems; i++)
            {
                stockPredictor.CalculateOutputs(testTargetInputs[i - 1]);
                double[] realOutputs = stockPredictor.GetOutputs();
                double sumError = 0.0;
                for (int k = 0; k < output; k++)
                {
                    sumError += Math.Pow(realOutputs[k] - testTargetOutputs[i - 1][k], 2);

                }
                totalError += sumError;
            }
            totalError /= numberOfTestItems;
            Console.WriteLine("The overall error is: " + totalError);
            Console.ReadLine();
            Console.WriteLine();
            Console.WriteLine("Do you wish to save this? Y/N");
            string choice = Console.ReadLine().ToLower();

            if (choice == "y")
            {
                Console.Write("Filename (without extension): ");
                string filenameForNetwork = Console.ReadLine() + ".nn";
                stockPredictor.SaveNetToFile(filenameForNetwork);
            }

            Console.WriteLine("Press enter to return to the menu...");
            Console.ReadLine();

        }
        public static List<StockData> LoadStockMarketData(string filename)
        {
            List<StockData> AllTheData = new List<StockData>();

            StreamReader reader = null;
            try
            {
                reader = new StreamReader(filename);

                reader.ReadLine(); //ignore the first line, as this just states headings.

                do
                {
                    string recordText = reader.ReadLine();
                   // Console.WriteLine("REad text: " + recordText);
                   // Console.ReadLine();
                    StockData record = new StockData();
                    string[] recordTextArray = recordText.Split(',');
                    //Console.WriteLine("Attempting to parse: " + recordTextArray[0] + " with value " + recordTextArray[1]);
                    record.Date = DateTime.Parse(recordTextArray[0]);
                    record.Value = Double.Parse(recordTextArray[1]);
                    AllTheData.Add(record);
                 //   Console.ReadLine();

                } while (!reader.EndOfStream);
            }
            catch (Exception ex)
            {

                //throw ex;
            }
            finally
            {
                if (reader != null)
                    reader.Close();

            }
            return AllTheData;
        }

        //NB: this isn't actually used, as we aren't taking stock markets into account. Remove all trace of this from teh research (keep backup though)
        public static void LoadInterestRateData(string filename,ref Dictionary<DateTime,Double> data)
        {
            Dictionary<DateTime, Double> interestRateData = new Dictionary<DateTime, Double>();

            StreamReader reader = null;
            try
            {
                reader = new StreamReader(filename);

                do
                {
                    string[] A = reader.ReadLine().Split(',');
                    interestRateData.Add(DateTime.Parse(A[0]), double.Parse(A[1]));

                } while (!reader.EndOfStream);



                
            }
            catch (Exception ex)
            {

                throw ex;
            }
            finally
            {
                if (reader != null)
                    reader.Close();

            }
           
        }

        public static void LoadNetworkFromFile()
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Enter filename:");
            string filename = Console.ReadLine();
            stockPredictor = new NeuralNetwork(filename);

            if (stockPredictor == null)
            {
                Console.WriteLine("An error occured. Was the filename correct?");
                Console.ReadLine();
                return;
            }
            else
                Console.WriteLine("Loaded. Press enter to continue...");

            Console.ReadLine();

        }

        public static double GetRawInput()
        {
            Console.Write("Input: ");
            string inputStr;
            double inputNum;

            do 
	        {       
	                inputStr = Console.ReadLine();
	        } while (Double.TryParse(inputStr, out inputNum));

            return Double.Parse(inputStr);

        }


        public static void PredictSpecificValues()
        {
            if (stockPredictor == null) return;

            Console.WriteLine("Enter the raw stock market values, separated by new lines:");
            Console.WriteLine();

            double[] inputs = new double[stockPredictor.InputNeuronCount()];
            for (int i = 0; i < stockPredictor.InputNeuronCount(); i++)
            {
                
                inputs[i] = GetRawInput();
            }
            Console.WriteLine();
            stockPredictor.CalculateOutputs(inputs);
            Console.WriteLine("Outputs:");
            double[] outputs = stockPredictor.GetOutputs();

            for (int i = 0; i < stockPredictor.OutputNeuronCount(); i++)
                Console.WriteLine(outputs[i]);

            Console.WriteLine();
            Console.WriteLine("Press enter to continue...");
            Console.ReadLine();
        }
        public static void TestNetworkAgainstFile(NeuralNetwork stockPredictor)
        {
            Console.WriteLine();
            Console.WriteLine("Enter filename of the test data:");
            string filename = Console.ReadLine();

            List<StockData> stockMarketData = LoadStockMarketData(filename);

            List<double[]> targetInputs = new List<double[]>();
            List<double[]> targetOutputs = new List<double[]>();

            Normalize(targetInputs, targetOutputs, stockMarketData);
            double numberOfCorrectSigns = 0;
            double globalError = 0.0;
            int i = 0;
            for (i = 0; i < targetInputs.Count; i++)
            {
                double epochError = 0.0;
                Console.WriteLine("Inputs: ");
                stockPredictor.CalculateOutputs(targetInputs[i]);
                double[] outputs = stockPredictor.GetOutputs();

                for (int j = 0; j < targetInputs[0].Length; j++)
                    Console.Write(targetInputs[i][j] + ",");
                
                Console.WriteLine("Outputs: ");

                for (int k = 0; k < targetOutputs[0].Length; k++)
                {
                    Console.WriteLine(targetOutputs[i][k] + "," + outputs[k] );
                    epochError += Math.Pow((targetOutputs[i][k] - outputs[k]),2);
                    if (!((targetOutputs[i][k] >= 0) ^ (outputs[k] >= 0)))
                        numberOfCorrectSigns++;
                }
                globalError += epochError;
            }
            Console.WriteLine("-----------------------------------");
            Console.WriteLine("Actual error is: " + globalError);
            Console.WriteLine();
            Console.WriteLine("Sign was correct " + numberOfCorrectSigns + " out of " + (i + 1) + " times.");
            double percent = (numberOfCorrectSigns / (i + 1)) * 100;
            Console.WriteLine("This gives a percentage sign correctness of: " + Math.Round(percent, 2) + "%");
            Console.ReadLine();

        }


        static void Main(string[] args)
        {
            string choice = "";

            do
            {

                Console.ForegroundColor = ConsoleColor.Gray;
                Console.WriteLine("Welcome to the ANN Stock Predictor");
                Console.WriteLine("Copyright Ryan Collins 2014");
                Console.WriteLine();
                Console.WriteLine("Select an option:");
                Console.WriteLine("1.  Create and train a new network");
                Console.WriteLine("2.  Load a network from file");
                Console.WriteLine("");
                Console.WriteLine("3.  Test network against a file");
                Console.WriteLine("4.  Predict!");
                Console.WriteLine("");
                Console.WriteLine("5.  About the program");
                Console.WriteLine("Press q to exit");
                Console.Write("Selection: ");
                choice = Console.ReadLine();

                switch (choice)
                {
                    case "1":
                        Train();
                        break;
                    case "2":
                        LoadNetworkFromFile();
                        break;
                    case "3":
                        if (stockPredictor != null)
                        {
                            TestNetworkAgainstFile(stockPredictor);
                        }
                        break;
                    case "4":
                        PredictSpecificValues();
                        break;
                    case "5":
                        About();
                        break;
                    default:
                        break;
                }
                Console.Clear();
            } while (choice != "q");
        }

        private static void About()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("About:");
            Console.WriteLine("");
            Console.WriteLine("Copyright Ryan Collins 2014");
            Console.WriteLine("Last Updated: 10/10/2014");
            Console.WriteLine("");
            Console.WriteLine("This was designed as part of my EPQ project to evaluate");
            Console.WriteLine("the effectiveness of using ANNs at predicting stock market indices.");
            Console.WriteLine("");
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine("Press enter to continue...");
            Console.ReadLine();
            Console.Clear();
        }


      public class StockData
        {
            public DateTime Date;
            public double Value;
            public double Change;
            
        }

    }
}

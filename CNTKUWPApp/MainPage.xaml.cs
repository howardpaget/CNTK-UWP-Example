using CNTKLib;
using Windows.UI.Xaml.Controls;

namespace CNTKUWPApp
{
    public sealed partial class MainPage : Page
    {
        private CNTKModel Model;

        public MainPage()
        {
            InitializeComponent();

            // Load the model create by CNTKFitExample.py
            Model = new CNTKModel("model.model");

            UpdateOutputBox();
        }

        private void InputValue_TextChanged(object sender, TextChangedEventArgs e)
        {
            UpdateOutputBox();
        }

        private void UpdateOutputBox()
        {
            try
            {
                var input1 = float.Parse(InputValue1.Text);
                var input2 = float.Parse(InputValue2.Text);

                // Display the model output in the output text box
                ModelOutput.Text = $"Model output: {Model.GetPrediction(input1, input2)}";
            }
            catch
            {
                ModelOutput.Text = "Input values must be numbers.";
            }
        }
    }
}

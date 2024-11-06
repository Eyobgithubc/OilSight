import React, { useState } from "react";
import axios from "axios";

function App() {
  // State to store input prices and predicted prices
  const [inputPrices, setInputPrices] = useState("");
  const [predictedPrices, setPredictedPrices] = useState([]);
  const [error, setError] = useState("");

  // Function to handle the form submit and make a POST request to Flask
  const handlePredict = async (event) => {
    event.preventDefault();

    // Convert the comma-separated string of prices into an array of numbers
    const pricesArray = inputPrices.split(',').map(price => parseFloat(price.trim()));

    if (pricesArray.length < 60) {
      setError("Please enter at least 60 prices.");
      return;
    }

    try {
      // Make a POST request to the Flask API
      const response = await axios.post("http://localhost:5000/predict", {
        prices: pricesArray,
      });

      // Set the predicted prices returned by the Flask API
      setPredictedPrices(response.data.predicted_prices);
      setError("");  // Clear any previous errors
    } catch (error) {
      console.error("Error making prediction:", error);
      setError("Error with the prediction request.");
      setPredictedPrices([]);
    }
  };

  return (
    <div className="App">
      <h1>Brent Oil Price Prediction</h1>

      <form onSubmit={handlePredict}>
        <label htmlFor="prices">
          Enter past oil prices (comma-separated):
        </label>
        <input
          type="text"
          id="prices"
          name="prices"
          value={inputPrices}
          onChange={(e) => setInputPrices(e.target.value)}
          placeholder="Enter prices like 70, 72, 73, 74..."
        />
        <button type="submit">Predict</button>
      </form>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {predictedPrices.length > 0 && (
        <div>
          <h2>Predicted Prices:</h2>
          <ul>
            {predictedPrices.map((price, index) => (
              <li key={index}>${price.toFixed(2)}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;

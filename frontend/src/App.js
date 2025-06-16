import React, { useState } from "react";
import { useForm } from "react-hook-form";
import {
  Button,
  Container,
  TextField,
  Grid,
  Card,
  CardContent,
  Typography,
  MenuItem,
  FormControl,
  InputLabel,
  Select,
  Alert,
  CircularProgress,
  Box
} from "@mui/material";
import * as yup from "yup";
import { yupResolver } from "@hookform/resolvers/yup";
import TopHeader from "./components/ui/TopHeader";
import Header from "./components/ui/Header";
import Footer from "./components/ui/Footer";

// Validation schema
const schema = yup.object().shape({
  state: yup.string().required("State is required"),
  query: yup.string().required("Query is required"),
});

// States list
const states = [
  "Andhra Pradesh",  "Andaman and Nicobar Islands","Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
  "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
  "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
  "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
  "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
];

function App() {
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [aiResponse, setAiResponse] = useState(null);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm({
    resolver: yupResolver(schema),
  });

  // Simple gibberish detector
  const isGibberish = (text) => {
    const repeatedChars = /(.)\1{4,}/; // Repeated characters
    const words = text.trim().split(/\s+/);
    return words.length < 3 || repeatedChars.test(text);
  };

  const onSubmit = async (data) => {
    setIsLoading(true);
    setResponse(null);
    setAiResponse(null);

    if (isGibberish(data.query)) {
      setAiResponse("Your query seems unclear. Please write a meaningful question.");
      setIsLoading(false);
      return;
    }

    try {
      const res = await fetch("http://localhost:8000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          state: data.state,
          query: data.query,
        }),
      });

      const result = await res.json();
      setResponse(data);
      setAiResponse(result.answer);
    } catch (error) {
      console.error("Error:", error);
      setAiResponse("Something went wrong. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Box sx={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      {/* Header */}
      <TopHeader />
      <Header />

      {/* Main Content */}
      <Container sx={{ marginTop: "40px", flexGrow: 1 }}>
        <Grid container justifyContent="center">
          <Grid item xs={12} sm={8} md={6}>
            <Card sx={{ boxShadow: 3, borderRadius: 2 }}>
              <CardContent>
                <Typography variant="h6" textAlign="center" mb={2}>
                  Water Scheme Query (Select a state and ask your questions!)
                </Typography>
                <form onSubmit={handleSubmit(onSubmit)}>
                  {/* Select State */}
                  <FormControl
                    fullWidth
                    error={!!errors.state}
                    sx={{ marginBottom: "16px" }}
                  >
                    <InputLabel>Select State</InputLabel>
                    <Select
                      label="Select State"
                      {...register("state")}
                      defaultValue=""
                    >
                      {states.map((state, index) => (
                        <MenuItem key={index} value={state}>
                          {state}
                        </MenuItem>
                      ))}
                    </Select>
                    <Typography variant="body2" color="error">
                      {errors.state?.message}
                    </Typography>
                  </FormControl>

                  {/* Query Input */}
                  <TextField
                    label="Enter Your Query"
                    variant="outlined"
                    fullWidth
                    multiline
                    rows={4}
                    {...register("query")}
                    error={!!errors.query}
                    helperText={errors.query?.message}
                    sx={{ marginBottom: "16px" }}
                  />

                  {/* Submit Button */}
                  <Button
                    variant="contained"
                    color="primary"
                    type="submit"
                    fullWidth
                    sx={{ padding: "12px" }}
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <CircularProgress size={24} color="inherit" />
                    ) : (
                      "Submit Query"
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Container>

      {/* Response Display */}
      {response && (
        <Container sx={{ marginTop: "40px" }}>
          <Grid container justifyContent="center">
            <Grid item xs={12} sm={8} md={6}>
              <Alert severity="success" sx={{ boxShadow: 3 }}>
                <Typography variant="h6">
                  Query Submitted Successfully!
                </Typography>
                <Typography>
                  <strong>State:</strong> {response.state}
                </Typography>
                <Typography>
                  <strong>Your Query:</strong> {response.query}
                </Typography>
                <Typography variant="h6" mt={2}>
                  <strong>AI Response:</strong>
                </Typography>
                <Typography>{aiResponse}</Typography>
              </Alert>
            </Grid>
          </Grid>
        </Container>
      )}

      {/* Footer */}
      <Footer />
    </Box>
  );
}

export default App;

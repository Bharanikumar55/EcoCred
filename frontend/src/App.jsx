import React from "react";
import LoanForm from "./components/LoanForm";
import Chatbot from "./components/Chatbot";
import "./App.css";

export default function App() {
  return (
    <div className="app-root">
      <header className="app-header">
        <h1>EcoCred — Loan Eligibility & Explainability</h1>
      </header>

      <main className="app-main">
        <section className="form-section">
          <LoanForm />
        </section>
      </main>

      {/* Floating Chatbot */}
      <Chatbot />
    </div>
  );
}

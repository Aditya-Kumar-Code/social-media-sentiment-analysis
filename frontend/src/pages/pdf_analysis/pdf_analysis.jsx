// src/components/PdfViewer.js
import React, { useState, useEffect } from "react";
import axios from "axios";
import { Worker, Viewer } from "@react-pdf-viewer/core";
import { defaultLayoutPlugin } from "@react-pdf-viewer/default-layout";
import "@react-pdf-viewer/core/lib/styles/index.css";
import "@react-pdf-viewer/default-layout/lib/styles/index.css";

const PdfAnalysis = () => {
  const [pdfData, setPdfData] = useState(null);
  const defaultLayout = defaultLayoutPlugin();

  useEffect(() => {
    const fetchPdf = async () => {
      try {
        const response = await axios.post(
          "http://127.0.0.1:8000/pdf_generation/",
          {
            topics: ["brand24"],
            limit: 2,
          },
          {
            responseType: "arraybuffer",
          }
        );

        const blob = new Blob([response.data], { type: "application/pdf" });
        const url = URL.createObjectURL(blob);
        setPdfData(url);
      } catch (error) {
        console.error("Error fetching PDF:", error);
      }
    };

    fetchPdf();
  }, []);

  return (
    <div className="pdf-viewer">
      {pdfData ? (
        <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
          <Viewer fileUrl={pdfData} plugins={[defaultLayout]} />
        </Worker>
      ) : (
        <p>Loading PDF...</p>
      )}
    </div>
  );
};

export default PdfAnalysis;


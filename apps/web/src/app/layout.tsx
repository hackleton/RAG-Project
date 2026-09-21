import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Healthcare Intelligence Copilot",
  description:
    "Grounded question answering over synthetic healthcare documents and structured data.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-surface text-ink antialiased">{children}</body>
    </html>
  );
}

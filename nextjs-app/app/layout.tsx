import "@/styles/globals.css";

import { Toaster } from "@/components/ui/toaster";
import { siteConfig } from "@/config/site";
import { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import { cn } from "@/lib/utils";

// Use local copy to avoid having NextJS fetch the file on the Internet during
// build time
const inter = localFont({
  src: "./InterVariable.ttf",
});

export const metadata: Metadata = {
  title: siteConfig.name,
  description: siteConfig.description,
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "black" },
  ],
};

interface RootLayoutProps {
  children: React.ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body
        className={cn(
          "min-h-screen bg-background antialiased bg-gradient-to-b from-[#1B2538] to-[#0F172A]",
          inter.className
        )}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}

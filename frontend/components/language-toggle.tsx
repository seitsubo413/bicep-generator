"use client"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface LanguageToggleProps {
  currentLanguage: "ja" | "en"
  onLanguageChange: (language: "ja" | "en") => void
}

export function LanguageToggle({ currentLanguage, onLanguageChange }: LanguageToggleProps) {
  return (
    <div className="flex items-center bg-muted/60 dark:bg-muted rounded-lg p-1">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onLanguageChange("ja")}
        className={cn(
          "px-3 py-1.5 text-sm font-medium transition-all",
          currentLanguage === "ja"
            ? "bg-background text-foreground shadow-sm"
            : "text-foreground/70 hover:text-foreground",
        )}
      >
        日本語
      </Button>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => onLanguageChange("en")}
        className={cn(
          "px-3 py-1.5 text-sm font-medium transition-all",
          currentLanguage === "en"
            ? "bg-background text-foreground shadow-sm"
            : "text-foreground/70 hover:text-foreground",
        )}
      >
        English
      </Button>
    </div>
  )
}
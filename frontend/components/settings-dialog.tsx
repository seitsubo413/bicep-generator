"use client"

import { Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { LanguageToggle } from "./language-toggle"
import { useTranslation } from "react-i18next"

interface SettingsDialogProps {
  onLanguageChange: (language: "ja" | "en") => void
}

export function SettingsDialog({ onLanguageChange }: SettingsDialogProps) {
  const { i18n } = useTranslation()
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-xl transition-all duration-200"
        >
          <Settings className="h-4 w-4" />
          <span className="sr-only">Settings</span>
        </Button>
      </DialogTrigger>
      <DialogContent className="rounded-2xl border-border/30 bg-card/95 backdrop-blur-sm">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text">
            Settings
          </DialogTitle>
        </DialogHeader>
        <div className="space-y-6 pt-2">
          <div className="flex items-center justify-between p-4 rounded-xl bg-muted/30 border border-border/20">
            <span className="font-medium">Language</span>
            <LanguageToggle currentLanguage={i18n.language as "ja" | "en"} onLanguageChange={onLanguageChange} />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
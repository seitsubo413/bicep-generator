"use client"

import React from "react"
import { useTranslation } from "react-i18next"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Settings } from "lucide-react"
import { LanguageToggle } from "@/components/language-toggle"

interface SettingsDialogProps {
  onLanguageChange?: (language: "ja" | "en") => void
}

export function SettingsDialog({ onLanguageChange }: SettingsDialogProps) {
  const { t, i18n } = useTranslation()

  const handleLanguageChange = (newLanguage: "ja" | "en") => {
    onLanguageChange?.(newLanguage)
  }

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button 
          variant="ghost" 
          size="sm" 
          className="text-muted hover:text-app hover-bg-surface-2"
          title={t("ui.editor.settings_tooltip")}
        >
          <Settings className="h-4 w-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>{t("ui.settings.title")}</DialogTitle>
          <DialogDescription>
            {t("ui.settings.description")}
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid grid-cols-4 items-center gap-4">
            <Label htmlFor="language" className="text-right">
              {t("ui.settings.language_label")}
            </Label>
            <div className="col-span-3">
              <LanguageToggle 
                currentLanguage={i18n.language as "ja" | "en"} 
                onLanguageChange={handleLanguageChange}
              />
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
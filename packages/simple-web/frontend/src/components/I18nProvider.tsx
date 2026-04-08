"use client";

import { createContext, useContext, useState, ReactNode } from "react";
import ko from "../../messages/ko.json";
import en from "../../messages/en.json";

type Messages = typeof ko;
type Locale = "ko" | "en";

const messages: Record<Locale, Messages> = { ko, en };

const I18nContext = createContext<{
  t: Messages;
  locale: Locale;
  setLocale: (l: Locale) => void;
}>({ t: ko, locale: "ko", setLocale: () => {} });

export function I18nProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState<Locale>("ko");
  return (
    <I18nContext.Provider value={{ t: messages[locale], locale, setLocale }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  return useContext(I18nContext);
}

export function LanguageToggle() {
  const { locale, setLocale } = useI18n();
  return (
    <button
      onClick={() => setLocale(locale === "ko" ? "en" : "ko")}
      className="px-3 py-1 rounded-md text-xs font-medium transition-colors"
      style={{
        background: "var(--surface-elevated)",
        color: "var(--surface-muted)",
        border: "1px solid var(--surface-border)",
      }}
    >
      {locale === "ko" ? "EN" : "KO"}
    </button>
  );
}

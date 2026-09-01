const base = {
  fontSans: "'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif",
  radiusSm: '10px',
  radiusMd: '16px',
  radiusLg: '24px',
  spaceUnit: '4px',
};

export const THEMES = Object.freeze({
  light: { label: 'Light', colorScheme: 'light', colors: { bg: '#f4f6f8', surface: '#ffffff', raised: '#eef1f5', text: '#17202a', muted: '#5d6b78', border: '#d8e0e8', accent: '#2563eb', accentStrong: '#1d4ed8', accentSoft: '#dbeafe', success: '#16804b', danger: '#c53030', shadow: '0 18px 45px rgba(27, 43, 64, .12)' } },
  dark: { label: 'Dark', colorScheme: 'dark', colors: { bg: '#0b0d12', surface: '#121720', raised: '#1a2230', text: '#f2f5f8', muted: '#9ca8b6', border: '#293444', accent: '#7aa2ff', accentStrong: '#a4bdff', accentSoft: '#1f3156', success: '#5ed39a', danger: '#ff8585', shadow: '0 20px 60px rgba(0, 0, 0, .32)' } },
  professional: { label: 'Professional', colorScheme: 'dark', colors: { bg: '#101820', surface: '#17232d', raised: '#20313d', text: '#edf4f7', muted: '#9cb0ba', border: '#344954', accent: '#56b5c8', accentStrong: '#8bd6e1', accentSoft: '#173d48', success: '#64c59a', danger: '#f18b83', shadow: '0 18px 50px rgba(0, 18, 28, .3)' } },
  modern: { label: 'Modern', colorScheme: 'dark', colors: { bg: '#11101b', surface: '#1a1828', raised: '#28223c', text: '#f7f2ff', muted: '#b8acc9', border: '#3c3453', accent: '#b38cff', accentStrong: '#d0baff', accentSoft: '#382b59', success: '#66d7ae', danger: '#ff8ea8', shadow: '0 18px 55px rgba(38, 18, 72, .34)' } },
  minimal: { label: 'Minimal', colorScheme: 'light', colors: { bg: '#fafafa', surface: '#ffffff', raised: '#f1f1f1', text: '#202124', muted: '#70757a', border: '#e1e1e1', accent: '#202124', accentStrong: '#000000', accentSoft: '#e9e9e9', success: '#2e7d5b', danger: '#b23b3b', shadow: '0 12px 30px rgba(32, 33, 36, .08)' } },
  gaming: { label: 'Gaming', colorScheme: 'dark', colors: { bg: '#080b12', surface: '#101522', raised: '#182039', text: '#eef6ff', muted: '#92a4bd', border: '#263653', accent: '#42f5b9', accentStrong: '#8affd5', accentSoft: '#123f3b', success: '#42f5b9', danger: '#ff5c8a', shadow: '0 18px 55px rgba(0, 255, 188, .1)' } },
  anime: { label: 'Anime', colorScheme: 'light', colors: { bg: '#fff5fb', surface: '#ffffff', raised: '#ffe7f3', text: '#321d37', muted: '#876c83', border: '#f2c9df', accent: '#e85ca5', accentStrong: '#c83b86', accentSoft: '#ffd8eb', success: '#39a884', danger: '#d94e67', shadow: '0 18px 50px rgba(196, 77, 143, .14)' } },
});

export function themeVariables(themeId) {
  const theme = THEMES[themeId] || THEMES.dark;
  const { colors } = theme;
  return {
    '--font-sans': base.fontSans,
    '--radius-sm': base.radiusSm,
    '--radius-md': base.radiusMd,
    '--radius-lg': base.radiusLg,
    '--bg': colors.bg,
    '--surface': colors.surface,
    '--raised': colors.raised,
    '--text': colors.text,
    '--muted': colors.muted,
    '--border': colors.border,
    '--accent': colors.accent,
    '--accent-strong': colors.accentStrong,
    '--accent-soft': colors.accentSoft,
    '--success': colors.success,
    '--danger': colors.danger,
    '--shadow': colors.shadow,
  };
}

// Adapted from Utkarsh Verma's SVG Spinners (MIT).
// https://github.com/n3r4zzurr0/svg-spinners/blob/main/svg-css/3-dots-rotate.svg
// License retained in ../assets/svg-spinners/LICENSE.
export function AnalysisLoader() {
  return (
    <svg
      className="analysis-loader"
      width="32"
      height="32"
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      <circle cx="12" cy="12" r="3" />
      <g className="analysis-loader-orbit">
        <circle cx="4" cy="12" r="3" />
        <circle cx="20" cy="12" r="3" />
      </g>
    </svg>
  );
}

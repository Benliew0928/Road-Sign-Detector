import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import LiveCameraWallApp from "./LiveCameraWallApp";
import PhoneCameraApp from "./PhoneCameraApp";
import "./styles.css";
import "./night.css";

const appPath = window.location.pathname.replace(/\/+$/, "") || "/";
const isPhoneRoute = appPath === "/phone";
const isLiveWallRoute = appPath === "/live";
document.body.classList.toggle("phone-route", isPhoneRoute);
document.body.classList.toggle("live-wall-route", isLiveWallRoute);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {isPhoneRoute ? <PhoneCameraApp /> : isLiveWallRoute ? <LiveCameraWallApp /> : <App />}
  </StrictMode>,
);

import "./lens.css";

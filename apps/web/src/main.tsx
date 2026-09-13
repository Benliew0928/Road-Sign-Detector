import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
import LiveCameraWallApp from "./LiveCameraWallApp";
import PhoneCameraApp from "./PhoneCameraApp";
import "./styles.css";
import "./night.css";

const isPhoneRoute = window.location.pathname === "/phone";
const isLiveWallRoute = window.location.pathname === "/live";
document.body.classList.toggle("phone-route", isPhoneRoute);
document.body.classList.toggle("live-wall-route", isLiveWallRoute);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {isPhoneRoute ? <PhoneCameraApp /> : isLiveWallRoute ? <LiveCameraWallApp /> : <App />}
  </StrictMode>,
);

import "./lens.css";

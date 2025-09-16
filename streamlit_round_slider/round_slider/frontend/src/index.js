import React, { useEffect, useRef } from "react"
import { withStreamlitConnection, Streamlit } from "streamlit-component-lib"
import "round-slider" // import roundSlider library
import "round-slider/dist/roundslider.min.css" // import the CSS

const AngleSlider = () => {
  const sliderRef = useRef(null)

  useEffect(() => {
    if (!sliderRef.current) return

    // Initialize roundSlider
    window.$(sliderRef.current).roundSlider({
      radius: 100,
      width: 16,
      handleSize: "+12",
      handleShape: "dot",
      sliderType: "min-range",
      value: 0,
      max: 360,
      min: 0,
      step: 1,
      circleShape: "pie",
      startAngle: 315,
      drag: function (args) {
        Streamlit.setComponentValue(args.value)
      },
      change: function (args) {
        Streamlit.setComponentValue(args.value)
      },
    })
  }, [])

  return <div id="slider" ref={sliderRef}></div>
}

export default withStreamlitConnection(AngleSlider)

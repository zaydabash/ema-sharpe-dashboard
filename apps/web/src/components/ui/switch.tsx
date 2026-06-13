import * as React from "react"
import { cn } from "@/lib/utils"

export interface SwitchProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size"> {
  size?: "sm" | "md" | "lg"
}

const Switch = React.forwardRef<HTMLInputElement, SwitchProps>(
  ({ className, size = "md", ...props }, ref) => {
    return (
      <label className="inline-flex items-center cursor-pointer">
        <input
          type="checkbox"
          className="sr-only"
          ref={ref}
          {...props}
        />
        <div
          className={cn(
            "relative rounded-full border border-border bg-secondary transition-colors duration-200 ease-in-out",
            {
              "w-8 h-4": size === "sm",
              "w-11 h-6": size === "md",
              "w-14 h-7": size === "lg",
            },
            props.checked && "border-primary bg-primary",
            className
          )}
        >
          <div
            className={cn(
              "absolute top-0.5 left-0.5 rounded-full transition-transform duration-200 ease-in-out",
              props.checked ? "bg-primary-foreground" : "bg-muted-foreground",
              {
                "w-3 h-3": size === "sm",
                "w-5 h-5": size === "md",
                "w-6 h-6": size === "lg",
              },
              props.checked && {
                "translate-x-4": size === "sm",
                "translate-x-5": size === "md",
                "translate-x-7": size === "lg",
              }
            )}
          />
        </div>
      </label>
    )
  }
)
Switch.displayName = "Switch"

export { Switch }

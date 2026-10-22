import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "cn"
import { Slot } from "radix-ui"

const buttonVariants = cva(
  "group/button inline-flex shrink-0 items-center justify-center rounded-3xl border border-transparent bg-clip-padding py-[4pt] text-base font-medium whitespace-nowrap transition-all outline-none select-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 active:not-aria-[haspopup]:translate-y-px disabled:pointer-events-none disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default: "bg-primary text-primary-foreground hover:bg-primary-hover hover:[&:not([data-size^=icon])]:text-[color-mix(in_oklch,var(--primary-foreground),transparent_20%)]",
        outline:
          "border-border bg-transparent hover:bg-black/5 hover:text-foreground hover:[&:not([data-size^=icon])]:text-[color-mix(in_oklch,var(--foreground),transparent_20%)] aria-expanded:text-foreground dark:border-input dark:hover:bg-white/5",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-[color-mix(in_oklch,var(--secondary),black_8%)] hover:[&:not([data-size^=icon])]:text-[color-mix(in_oklch,var(--secondary-foreground),transparent_20%)] aria-expanded:bg-secondary aria-expanded:text-secondary-foreground",
        ghost:
          "hover:bg-black/5 hover:text-foreground hover:[&:not([data-size^=icon])]:text-[color-mix(in_oklch,var(--foreground),transparent_20%)] aria-expanded:bg-muted aria-expanded:text-foreground dark:hover:bg-white/5",
        destructive:
          "bg-destructive/10 text-destructive hover:bg-destructive/20 hover:[&:not([data-size^=icon])]:text-[color-mix(in_oklch,var(--destructive),transparent_20%)] focus-visible:border-destructive/40 focus-visible:ring-destructive/20 dark:bg-destructive/20 dark:hover:bg-black/20 dark:focus-visible:ring-destructive/40",
        link: "text-primary underline-offset-4 hover:text-primary-hover hover:[&:not([data-size^=icon])]:text-[color-mix(in_oklch,var(--primary),transparent_20%)] hover:underline",
      },
      size: {
        default:
          "h-[calc(2.5rem+8pt)] gap-2 px-[calc(1rem+4pt)] has-data-[icon=inline-end]:pr-[calc(.75rem+4pt)] has-data-[icon=inline-start]:pl-[calc(.75rem+4pt)]",
        xs: "h-[calc(2rem+8pt)] gap-1.5 rounded-3xl px-[calc(.75rem+4pt)] text-[calc(0.75rem+2pt)] in-data-[slot=button-group]:rounded-3xl has-data-[icon=inline-end]:pr-[calc(.625rem+4pt)] has-data-[icon=inline-start]:pl-[calc(.625rem+4pt)] [&_svg:not([class*='size-'])]:size-3",
        sm: "h-[calc(2.25rem+8pt)] gap-1.5 rounded-3xl px-[calc(.875rem+4pt)] text-[calc(0.8rem+2pt)] in-data-[slot=button-group]:rounded-3xl has-data-[icon=inline-end]:pr-[calc(.75rem+4pt)] has-data-[icon=inline-start]:pl-[calc(.75rem+4pt)] [&_svg:not([class*='size-'])]:size-3.5",
        lg: "h-[calc(2.75rem+8pt)] gap-2 px-[calc(1.25rem+4pt)] has-data-[icon=inline-end]:pr-[calc(1rem+4pt)] has-data-[icon=inline-start]:pl-[calc(1rem+4pt)]",
        icon: "size-[calc(2.5rem+8pt)]",
        "icon-xs":
          "size-[calc(2rem+8pt)] rounded-3xl in-data-[slot=button-group]:rounded-3xl [&_svg:not([class*='size-'])]:size-3",
        "icon-sm":
          "size-[calc(2.25rem+8pt)] rounded-3xl in-data-[slot=button-group]:rounded-3xl",
        "icon-lg": "size-[calc(2.75rem+8pt)]",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }

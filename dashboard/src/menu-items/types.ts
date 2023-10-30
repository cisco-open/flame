import React from "react";

export interface MenuItem {
    title: string,
    type: string,
    id: string,
    url: string,
    icon?: React.ReactNode,
}
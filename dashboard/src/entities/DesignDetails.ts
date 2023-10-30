import { Design } from "./Design";

export interface DesignDetails extends Design {
    schema: Schema;
}

export interface Schema {
    channels: Channel[];
    name: string;
    description: string;
    roles: Role[];
}

export interface Channel {
    description: string;
    funcTags: any;
    groupBy: GroupBy;
    name: string;
    pair: string[];
    index?: number;
}

export interface FuncTags {
    aggregator?: string[];
    trainer?: string[];
}

export interface GroupBy {
    type: string;
    value: string[];
}

export interface Role {
    description: string;
    groupAssociation: GroupAssociation[];
    isDataConsumer: boolean;
    name: string;
    index?: number;
    replica?: number;
}

export interface GroupAssociation {
    [key: string]: string;
}

export interface MappedFuncTag { roleName: string, funcTags: {
    value: string,
    selected: boolean,
    disabled: boolean,
}[] };
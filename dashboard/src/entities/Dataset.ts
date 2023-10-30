export interface Dataset {
    id: string;
    userId: string;
    name: string;
    description: string;
    url: string;
    dataFormat: string;
    realm: string;
    computeId: string;
    isPublic: boolean;
}
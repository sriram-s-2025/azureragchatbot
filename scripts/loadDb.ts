import { MongoClient, Db, Collection, Document } from 'mongodb';
import { pipeline, FeatureExtractionPipeline } from '@huggingface/transformers';
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer"
import OpenAI from "openai";
import { getBearerTokenProvider, DefaultAzureCredential } from "@azure/identity";

import "dotenv/config";

import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const { MONGO_CONNECTION_STRING } = process.env;

const claudeNewData = [
   'https://www.anthropic.com/claude-fable-and-mythos-5-2',
   'https://www.anthropic.com/news/public-benefit-fund',
   'https://www.anthropic.com/research/early-progress-drug-development',
   'https://www.anthropic.com/claude-haiku-5-5',
   'https://www.anthropic.com/claude-opus-5-5',
   'https://www.anthropic.com/news/s1-sec',
   'https://support.claude.com/en/articles/18185620-use-personal-finance-in-claude',
   'https://support.claude.com/en/articles/18325154-use-claude-meetings',
   'https://support.claude.com/en/articles/17454997-get-started-with-claude-motion',
]


async function getEmbeddings(sentences: string[]): Promise<number[][]> {
  // 1. Load the ONNX-compatible version of all-mpnet-base-v2
  const extractor: FeatureExtractionPipeline = await pipeline(
    'feature-extraction',
    'Xenova/all-mpnet-base-v2',
    {
      // Optional: Use 'q8' for smaller download/faster CPU inference, or 'fp32' for full precision
      dtype: 'q8', 
    }
  );

  // 2. Generate embeddings with mean pooling and L2 normalization
  const output = await extractor(sentences, {
    pooling: 'mean',
    normalize: true,
  });

  // 3. Convert the returned Tensor into a standard 2D number array
  return output.tolist() as number[][];
}

const scrapePage = async (url: string): Promise<string> => {
    const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: {
            headless: true,
        },
        gotoOptions: {
            waitUntil: "domcontentloaded",
        },
        evaluate: async (page, browser) => {
            const result = await page.evaluate(() => document.body.innerHTML);
            await browser.close();
            return result;
        },
    });
    
    return (await loader.scrape())?.replace(/<[^>]*>?/gm, " ") ?? "";
};



async function main(): Promise<void> {
    // Connection string for Azure DocumentDB cluster
    const connectionString = MONGO_CONNECTION_STRING;
    if (!connectionString) {
        throw new Error("MONGO_CONNECTION_STRING is not set");
    }

    // Create a new client and connect to the server
    const client = new MongoClient(connectionString);

    try
    {
        await client.connect();

        await client.db("admin").command({ ping: 1 });
        console.log("Successfully connected to Azure DocumentDB");

        // Get database reference
        const database: Db = client.db("Docs");
        console.log(`Connected to database: ${database.databaseName}`);
        

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 512,
            chunkOverlap: 100,
        });

        // Create the vector index using database.command()
        const res = await database.command({
            createIndexes: "ClaudeDocs",
            indexes: [
                {
                    name: "vectorSearchIndex",
                    key: { "vectorContent": "cosmosSearch" },
                    cosmosSearchOptions: {
                        kind: "vector-ivf",
                        numLists: 100,
                        similarity: "IP",
                        dimensions: 768 // For sentence-transformers/all-mpnet-base-v2
                    }
                }
            ]
        });
        console.log("Vector index creation result:", res);

        // Get collection reference
        const collection: Collection<Document> = database.collection("ClaudeDocs");
        console.log("Connected to collection: ClaudeDocs");

        for (const url of claudeNewData) {
            const content = await scrapePage(url);
            const chunks = await splitter.splitText(content);

            const embeddings = await getEmbeddings(chunks);

            const documents = chunks.map((text, index) => ({
                text: text,
                vectorContent: embeddings[index], // Must match your index key name
                metadata: {
                    chunkIndex: index,
                    source: "claude-docs"
                }
            }));

            const result = await collection.insertMany(documents);
            console.log(`Inserted ${result.insertedCount} chunks from ${url} into ClaudeDocs`);
        }

        
    }
    catch (error)
    {
        console.error("Failed to connect to Azure DocumentDB:", error);
    }
    finally
    {
        await client.close();
    }
}

main().catch((error) => {
    console.error("Failed to load database:", error);
    process.exitCode = 1;
});
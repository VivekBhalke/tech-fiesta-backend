import { PrismaClient } from '@prisma/client';

// Create a singleton instance of Prisma Client
let prisma;

if (process.env.NODE_ENV === 'production') {
  // In production, create a new PrismaClient instance
  prisma = new PrismaClient();
} else {
  // In development, use a global variable to prevent multiple instances during hot reload
  if (!global.prisma) {
    global.prisma = new PrismaClient();
  }
  prisma = global.prisma;
}
export default prisma;

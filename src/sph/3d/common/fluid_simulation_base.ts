/**
 * =============================================================================
 * Base class for 3D Fluid Simulation Orchestrators
 * =============================================================================
 *
 * Holds all shared GPU resources, uniform buffers, CPU staging arrays, and
 * the common buffer-reset logic so each renderer-specific subclass only needs
 * to implement its own setup and render method.
 */

import type { SimState, SimConfig, SpawnData } from './types.ts';
import { createSpawnData } from './spawn.ts';
import { FluidBuffers, type FluidBuffersOptions } from './fluid_buffers.ts';
import {
  SpatialGrid,
  type SpatialGridUniforms,
} from './spatial_grid.ts';
import { FluidPhysics, type PhysicsUniforms } from './fluid_physics.ts';
import type { EnvironmentConfig } from './environment.ts';

export abstract class FluidSimulationBase<
  TConfig extends SimConfig & EnvironmentConfig,
> {
  protected device: GPUDevice;
  protected context: GPUCanvasContext;
  protected config: TConfig;

  // --- Subsystems (Modular) ---
  protected buffers!: FluidBuffers;
  protected physics: FluidPhysics;
  protected grid: SpatialGrid;

  protected state!: SimState;

  // --- Grid Configuration ---
  protected gridRes = { x: 0, y: 0, z: 0 };
  protected gridTotalCells = 0;

  // --- Interaction State ---
  protected isPicking = false;
  protected interactionPos = { x: 0, y: 0, z: 0 };

  // --- Uniform Buffers ---
  protected physicsUniforms!: PhysicsUniforms;
  protected gridUniforms!: SpatialGridUniforms;

  // --- CPU Staging Buffers ---
  protected computeData = new Float32Array(8);
  protected integrateData = new Float32Array(24);
  protected hashParamsData = new Float32Array(8);
  protected sortParamsData = new Uint32Array(8);
  protected scanParamsDataL0 = new Uint32Array(4);
  protected scanParamsDataL1 = new Uint32Array(4);
  protected scanParamsDataL2 = new Uint32Array(4);
  protected densityParamsData = new Float32Array(12);
  protected pressureParamsData = new Float32Array(16);
  protected viscosityParamsData = new Float32Array(12);

  constructor(device: GPUDevice, context: GPUCanvasContext, config: TConfig) {
    this.device = device;
    this.context = context;
    this.config = config;

    this.physics = new FluidPhysics(device);
    this.grid = new SpatialGrid(device);

    this.physicsUniforms = {
      external: device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      density: device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      pressure: device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      viscosity: device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      integrate: device.createBuffer({
        size: 96,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };

    this.gridUniforms = {
      hash: device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      sort: device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      scanL0: device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      scanL1: device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
      scanL2: device.createBuffer({
        size: 32,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    };
  }

  get particleCount(): number {
    return this.buffers.particleCount;
  }

  get simulationState(): SimState {
    return this.state;
  }

  /**
   * Recreates the shared GPU buffers and bind groups for physics and the
   * spatial grid. Subclass `reset()` methods should call this first, then
   * perform their own renderer-specific setup.
   *
   * @param bufferArgs - Extra options forwarded to the {@link FluidBuffers}
   *   constructor beyond the always-present `gridTotalCells`. Use this to
   *   opt-in to optional buffer allocations such as foam buffers.
   */
  protected resetBuffers(
    bufferArgs: Omit<FluidBuffersOptions, 'gridTotalCells'> = {}
  ): void {
    if (this.buffers) {
      this.buffers.destroy();
    }

    const { boundsSize, smoothingRadius } = this.config;
    this.gridRes = {
      x: Math.ceil(boundsSize.x / smoothingRadius),
      y: Math.ceil(boundsSize.y / smoothingRadius),
      z: Math.ceil(boundsSize.z / smoothingRadius),
    };
    this.gridTotalCells = this.gridRes.x * this.gridRes.y * this.gridRes.z;

    const spawn = createSpawnData(this.config);
    this.state = this.createStateFromSpawn(spawn);

    this.buffers = new FluidBuffers(this.device, spawn, {
      gridTotalCells: this.gridTotalCells,
      ...bufferArgs,
    });

    this.physics.createBindGroups(this.buffers, this.physicsUniforms);
    this.grid.createBindGroups(this.buffers, this.gridUniforms);
  }

  protected updatePrefixSumUniforms(): void {
    const blocksL0 = Math.ceil((this.gridTotalCells + 1) / 512);
    const blocksL1 = Math.ceil(blocksL0 / 512);
    this.scanParamsDataL0[0] = this.gridTotalCells + 1;
    this.scanParamsDataL1[0] = blocksL0;
    this.scanParamsDataL2[0] = blocksL1;
    this.device.queue.writeBuffer(
      this.gridUniforms.scanL0,
      0,
      this.scanParamsDataL0
    );
    this.device.queue.writeBuffer(
      this.gridUniforms.scanL1,
      0,
      this.scanParamsDataL1
    );
    this.device.queue.writeBuffer(
      this.gridUniforms.scanL2,
      0,
      this.scanParamsDataL2
    );
  }

  protected createStateFromSpawn(spawn: SpawnData): SimState {
    return {
      positions: spawn.positions,
      predicted: new Float32Array(spawn.positions),
      velocities: spawn.velocities,
      densities: new Float32Array(spawn.count * 2),
      keys: new Uint32Array(spawn.count),
      sortedKeys: new Uint32Array(spawn.count),
      indices: new Uint32Array(spawn.count),
      sortOffsets: new Uint32Array(spawn.count),
      spatialOffsets: new Uint32Array(spawn.count),
      positionsSorted: new Float32Array(spawn.count * 4),
      predictedSorted: new Float32Array(spawn.count * 4),
      velocitiesSorted: new Float32Array(spawn.count * 4),
      count: spawn.count,
      input: { worldX: 0, worldY: 0, worldZ: 0, pull: false, push: false },
    };
  }
}
